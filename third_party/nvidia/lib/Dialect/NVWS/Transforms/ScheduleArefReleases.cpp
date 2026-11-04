#include "lib/Dialect/TritonGPU/Transforms/WarpSpecialization/PartitionAttrs.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

#include <memory>
#include <optional>

namespace mlir::triton {

#define GEN_PASS_DEF_NVWSSCHEDULEAREFRELEASES
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"

namespace {
using namespace nvws;
namespace ttg = gpu;
namespace ttng = nvidia_gpu;

// Textual order is meaningful only within one partition and one pending
// pipeline stage/cluster. In particular, scalar uniformity does not identify
// dynamic instances in different partitions or iterations.
static bool sameContext(Operation *a, Operation *b) {
  if (!ttg::hasPartition(a) || !ttg::hasPartition(b))
    return false;
  auto aParts = ttg::getPartitionIds(a);
  auto bParts = ttg::getPartitionIds(b);
  if (aParts.size() != 1 || bParts.size() != 1 ||
      aParts.front() != bParts.front())
    return false;
  for (StringRef name : {"ttg.warp_specialize.tag", "loop.stage",
                         "loop.cluster"})
    if (a->getAttr(name) != b->getAttr(name))
      return false;
  return true;
}

static std::optional<AsyncOp> completionKind(Operation *exit) {
  auto kinds = exit->getAttrOfType<ArrayAttr>("async_ops");
  if (!kinds || kinds.size() != 1)
    return std::nullopt;
  auto kind = dyn_cast<AsyncOpAttr>(kinds[0]);
  if (!kind || (kind.getValue() != AsyncOp::NONE &&
                kind.getValue() != AsyncOp::TC5MMA))
    return std::nullopt;
  return kind.getValue();
}

// A lifetime can have several local segments: before the loop, at the start of
// its body, after the next acquire, and after the loop. A null begin means the
// block argument already owns this generation. end may be a loop/yield rather
// than a release; context supplies the partition and pending pipeline schedule.
struct Scope {
  Operation *begin;
  Operation *end;
  Operation *context;
  Operation *exit;
  ArefCreateOp channel;
  Value token;
  bool isGet;
  AsyncOp completion;
  DenseSet<Operation *> accesses;

  bool contains(Operation *op) const {
    if (op->getBlock() != end->getBlock() ||
        (begin && !begin->isBeforeInBlock(op)) || !op->isBeforeInBlock(end))
      return false;
    // Unpartitioned prologue/epilogue code executes outside the specialized
    // loop. It is checked for confinement, but is never a motion candidate.
    if (!ttg::hasPartition(context) && !ttg::hasPartition(op)) {
      for (StringRef name : {"ttg.warp_specialize.tag", "loop.stage",
                             "loop.cluster"})
        if (context->getAttr(name) != op->getAttr(name))
          return false;
      return true;
    }
    return sameContext(context, op);
  }
};

struct Channel {
  ArefCreateOp create;
  int64_t depth;
  bool carried = false;
  bool mma = false;
  SmallVector<std::unique_ptr<Scope>> scopes;
};

struct Enter {
  Operation *op;
  Value token;
  Value buffer;
  bool isGet;
};

// This is a recognizer for closed ownership protocols, not an assumption that
// Aref notifications publish only their own buffers. One producer and one
// consumer execute one acquire per logical generation. AssignStagePhase maps
// generation g to slot g % depth and toggles phase on wrap. It advances on
// ENTER, so the yielded token belongs to g+1, and the final drain to tripCount
// (including zero). Explicit stage/phase and conditional token merges are not
// certificates for this recognizer.
class Ownership {
public:
  Channel *certify(ArefCreateOp create) {
    auto found = channels.find(create);
    if (found != channels.end())
      return found->second.get();
    if (!attempted.insert(create).second || create.getBuffers().size() != 1)
      return nullptr;
    Value root = create.getBuffers()[0];
    auto alloc = root.getDefiningOp<ttng::TMEMAllocOp>();
    auto rootType = cast<ttg::MemDescType>(root.getType());
    if (!alloc || alloc->hasAttr("tensor_memory_col_offset") ||
        alloc->hasAttr("tensor_memory_row_offset") || !root.hasOneUse() ||
        rootType.getRank() != 3 || rootType.getShape()[0] < 1 ||
        rootType.getAllocShape() != rootType.getShape() ||
        (alloc.getToken() && !alloc.getToken().use_empty()))
      return nullptr;

    SmallVector<Enter> puts, gets;
    for (Operation *user : create->getUsers()) {
      Value token, stage, phase;
      SmallVector<Value> buffers;
      bool isGet;
      if (auto enter = dyn_cast<ArefGetEnterOp>(user)) {
        token = enter.getToken();
        stage = enter.getStage();
        phase = enter.getPhase();
        buffers.append(enter.getBuffers().begin(), enter.getBuffers().end());
        isGet = true;
      } else if (auto enter = dyn_cast<ArefPutEnterOp>(user)) {
        token = enter.getToken();
        stage = enter.getStage();
        phase = enter.getPhase();
        buffers.append(enter.getBuffers().begin(), enter.getBuffers().end());
        isGet = false;
      } else {
        continue;
      }
      if (stage || phase || buffers.size() != 1)
        return nullptr;
      // Enter verifies shape/element type, but not encoding or memory space.
      auto type = cast<ttg::MemDescType>(buffers[0].getType());
      if (type.getEncoding() != rootType.getEncoding() ||
          type.getMemorySpace() != rootType.getMemorySpace() ||
          type.getElementType() != rootType.getElementType() ||
          type.getShape() != rootType.getShape().drop_front() ||
          !validAllocationShape(type, rootType))
        return nullptr;
      (isGet ? gets : puts).push_back({user, token, buffers[0], isGet});
    }
    if (puts.empty() || gets.empty() || puts.size() > 2 || gets.size() > 2)
      return nullptr;

    auto channel = std::make_unique<Channel>();
    channel->create = create;
    channel->depth = rootType.getShape()[0];
    DenseSet<Operation *> recognized;
    if (puts.size() == 1 && gets.size() == 1) {
      auto loop = dyn_cast<scf::ForOp>(puts[0].op->getParentOp());
      if (!validLoopPair(loop, puts[0], gets[0]) ||
          !addDirect(*channel, puts[0], recognized) ||
          !addDirect(*channel, gets[0], recognized))
        return nullptr;
    } else if (puts.size() == 2 && gets.size() == 2) {
      if (!addCarried(*channel, puts, gets, recognized))
        return nullptr;
      channel->carried = true;
    } else {
      return nullptr;
    }
    for (Operation *user : create->getUsers())
      if (!recognized.contains(user))
        return nullptr;
    Channel *result = channel.get();
    channels[create] = std::move(channel);
    return result;
  }

  Scope *getScope(ArefGetExitOp exit) {
    auto create = exit.getAref().getDefiningOp<ArefCreateOp>();
    Channel *channel = create ? certify(create) : nullptr;
    if (!channel)
      return nullptr;
    for (auto &scope : channel->scopes)
      if (scope->exit == exit)
        return scope.get();
    return nullptr;
  }

  bool heldElsewhere(Operation *access, ArefGetExitOp exit, Scope &released) {
    Value descriptor;
    if (auto load = dyn_cast<ttng::TMEMLoadOp>(access))
      descriptor = load.getSrc();
    else if (auto store = dyn_cast<ttng::TMEMStoreOp>(access))
      descriptor = store.getDst();
    else
      // In particular, never cross an MMA, copy, commit or another Aref exit.
      // TC5 commits cover the prior issue group, not a descriptor-filtered set.
      return false;
    while (Operation *def = descriptor.getDefiningOp()) {
      if (def->hasTrait<OpTrait::MemDescViewTrait>()) {
        descriptor = def->getOperand(0);
        continue;
      }
      Value aref;
      if (auto enter = dyn_cast<ArefGetEnterOp>(def))
        aref = enter.getAref();
      else if (auto enter = dyn_cast<ArefPutEnterOp>(def))
        aref = enter.getAref();
      else if (auto buffer = dyn_cast<ArefBufferOp>(def))
        aref = buffer.getAref();
      else
        return false;
      auto create = aref.getDefiningOp<ArefCreateOp>();
      if (!create || create == released.channel)
        return false;
      Channel *channel = certify(create);
      if (!channel)
        return false;
      for (auto &scope : channel->scopes)
        if (scope->accesses.contains(access) && scope->contains(exit))
          return true;
      return false;
    }
    return false;
  }

  size_t size() const { return channels.size(); }
  unsigned multibufferCount() const {
    return llvm::count_if(channels, [](const auto &entry) {
      return entry.second->depth > 1;
    });
  }
  unsigned carriedCount() const {
    return llvm::count_if(channels, [](const auto &entry) {
      return entry.second->carried;
    });
  }
  unsigned mmaCount() const {
    return llvm::count_if(channels, [](const auto &entry) {
      return entry.second->mma;
    });
  }

private:
  static bool validAllocationShape(ttg::MemDescType type,
                                   ttg::MemDescType root) {
    return type.getAllocShape() == root.getShape() ||
           type.getAllocShape() == root.getShape().drop_front();
  }

  static bool validLoopPair(scf::ForOp loop, const Enter &put,
                            const Enter &get) {
    if (!loop || get.op->getBlock() != loop.getBody() ||
        put.op->getBlock() != loop.getBody() ||
        !loop->hasAttr("ttg.partition.stages") ||
        !loop->hasAttr("ttg.warp_specialize.tag") ||
        !ttg::hasPartition(put.op) || !ttg::hasPartition(get.op))
      return false;
    auto p = ttg::getPartitionIds(put.op), g = ttg::getPartitionIds(get.op);
    if (p.size() != 1 || g.size() != 1 || p.front() == g.front())
      return false;
    for (Operation *enter : {put.op, get.op})
      if (Attribute tag = enter->getAttr("ttg.warp_specialize.tag"))
        if (tag != loop->getAttr("ttg.warp_specialize.tag"))
          return false;
    return put.op->getAttr("ttg.warp_specialize.tag") ==
           get.op->getAttr("ttg.warp_specialize.tag");
  }

  // Prologue and drain are either serial code or the same owner as the body.
  static bool boundaryOwner(Operation *boundary, Operation *body,
                            scf::ForOp loop) {
    if (ttg::hasPartition(boundary)) {
      auto parts = ttg::getPartitionIds(boundary);
      if (parts.size() != 1 ||
          parts.front() != ttg::getPartitionIds(body).front())
        return false;
    } else if (!isa<triton::FuncOp>(loop->getParentOp())) {
      // Absence of a partition is only a serial boundary at function scope.
      // In particular, it is not a rendezvous in an enclosing WS loop.
      return false;
    }
    if (Attribute tag = boundary->getAttr("ttg.warp_specialize.tag"))
      if (tag != loop->getAttr("ttg.warp_specialize.tag"))
        return false;
    return !boundary->hasAttr("loop.stage") &&
           !boundary->hasAttr("loop.cluster");
  }

  static Operation *findExit(Value token, ArefCreateOp channel, bool isGet) {
    Operation *result = nullptr;
    for (Operation *user : token.getUsers()) {
      Value aref, stage;
      if (auto exit = dyn_cast<ArefGetExitOp>(user)) {
        if (!isGet)
          return nullptr;
        aref = exit.getAref();
        stage = exit.getStage();
      } else if (auto exit = dyn_cast<ArefPutExitOp>(user)) {
        if (isGet)
          return nullptr;
        aref = exit.getAref();
        stage = exit.getStage();
      } else if (isa<ArefBufferOp>(user)) {
        continue;
      } else {
        return nullptr;
      }
      if (result || aref != channel.getResult() || stage || !completionKind(user))
        return nullptr;
      result = user;
    }
    return result;
  }

  static bool addDirect(Channel &channel, const Enter &enter,
                        DenseSet<Operation *> &recognized) {
    Operation *exit = findExit(enter.token, channel.create, enter.isGet);
    if (!exit || exit->getBlock() != enter.op->getBlock() ||
        !enter.op->isBeforeInBlock(exit))
      return false;
    auto scope = std::make_unique<Scope>(Scope{
        enter.op, exit, enter.op, exit, channel.create, enter.token,
        enter.isGet, *completionKind(exit), {}});
    // Endpoints must have the same execution context, including unpartitioned
    // serial drains. contains deliberately excludes the endpoint itself.
    if (ttg::hasPartition(enter.op) || ttg::hasPartition(exit)) {
      if (!sameContext(enter.op, exit))
        return false;
    } else {
      for (StringRef name : {"ttg.warp_specialize.tag", "loop.stage", "loop.cluster"})
        if (enter.op->getAttr(name) != exit->getAttr(name))
          return false;
    }
    return addSegment(channel, std::move(scope), {enter.buffer}, exit,
                      recognized, enter.op);
  }

  static bool addCarried(Channel &channel, SmallVector<Enter> &puts,
                         SmallVector<Enter> &gets,
                         DenseSet<Operation *> &recognized) {
    // Find the unique direct loop operand use of the initial PUT token.
    Enter *initial = nullptr;
    scf::ForOp loop;
    unsigned index = 0;
    for (Enter &put : puts)
      for (OpOperand &use : put.token.getUses())
        if (auto candidate = dyn_cast<scf::ForOp>(use.getOwner())) {
          if (initial || use.getOperandNumber() < candidate.getNumControlOperands())
            return false;
          initial = &put;
          loop = candidate;
          index = use.getOperandNumber() - candidate.getNumControlOperands();
        }
    if (!initial)
      return false;
    Enter &next = &puts[0] == initial ? puts[1] : puts[0];
    Enter *bodyGet = nullptr, *drainGet = nullptr;
    for (Enter &get : gets) {
      if (get.op->getBlock() == loop.getBody()) {
        if (bodyGet)
          return false;
        bodyGet = &get;
      } else {
        drainGet = &get;
      }
    }
    if (!bodyGet || !drainGet || !validLoopPair(loop, next, *bodyGet) ||
        initial->op->getBlock() != loop->getBlock() ||
        !initial->op->isBeforeInBlock(loop) ||
        drainGet->op->getBlock() != loop->getBlock() ||
        !loop->isBeforeInBlock(drainGet->op) ||
        !boundaryOwner(initial->op, next.op, loop) ||
        !boundaryOwner(drainGet->op, bodyGet->op, loop))
      return false;
    // PartitionLoops assigns loop inputs and yielded values using the output
    // metadata, not the immediate token user's partition. The carried owner
    // must agree with that metadata before the token can be a certificate.
    auto outputs = loop->getAttrOfType<ArrayAttr>("ttg.partition.outputs");
    if (!outputs || outputs.size() != loop.getNumResults())
      return false;
    auto owner = dyn_cast<DenseI32ArrayAttr>(outputs[index]);
    if (!owner || owner.size() != 1 ||
        owner[0] != ttg::getPartitionIds(next.op).front() ||
        initial->buffer.getType() != next.buffer.getType())
      return false;
    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    if (yield.getOperand(index) != next.token)
      return false;
    Value incoming = loop.getRegionIterArg(index);
    Value outgoing = loop.getResult(index);
    Operation *bodyExit = findExit(incoming, channel.create, false);
    Operation *finalExit = findExit(outgoing, channel.create, false);
    if (!bodyExit || !finalExit || bodyExit->getBlock() != loop.getBody() ||
        !bodyExit->isBeforeInBlock(next.op) ||
        !sameContext(bodyExit, next.op) ||
        finalExit->getBlock() != loop->getBlock() ||
        !loop->isBeforeInBlock(finalExit) ||
        !finalExit->isBeforeInBlock(drainGet->op) ||
        !boundaryOwner(finalExit, next.op, loop) ||
        completionKind(bodyExit) != completionKind(finalExit))
      return false;
    AsyncOp completion = *completionKind(bodyExit);
    // The only forwarding uses are one initial operand and one matching yield.
    // addSegment checks all remaining token users and confines descriptors to
    // the local segment; no token/descriptor can bypass a generation boundary.
    auto pre = std::make_unique<Scope>(Scope{
        initial->op, loop, initial->op, nullptr, channel.create, initial->token,
        false, completion, {}});
    auto body = std::make_unique<Scope>(Scope{
        nullptr, bodyExit, bodyExit, bodyExit, channel.create, incoming,
        false, completion, {}});
    auto tail = std::make_unique<Scope>(Scope{
        next.op, yield, next.op, nullptr, channel.create, next.token,
        false, completion, {}});
    auto post = std::make_unique<Scope>(Scope{
        loop, finalExit, finalExit, finalExit, channel.create, outgoing,
        false, completion, {}});
    return addSegment(channel, std::move(pre), {initial->buffer}, loop,
                      recognized, initial->op) &&
           addSegment(channel, std::move(body), {}, bodyExit, recognized,
                      nullptr, initial->buffer.getType()) &&
           addSegment(channel, std::move(tail), {next.buffer}, yield,
                      recognized, next.op) &&
           addSegment(channel, std::move(post), {}, finalExit, recognized,
                      nullptr, initial->buffer.getType()) &&
           addDirect(channel, *bodyGet, recognized) &&
           addDirect(channel, *drainGet, recognized);
  }

  static bool addSegment(Channel &channel, std::unique_ptr<Scope> scope,
                         SmallVector<Value> buffers, Operation *tokenEnd,
                         DenseSet<Operation *> &recognized,
                         Operation *enter = nullptr, Type borrowedType = {}) {
    unsigned endUses = 0;
    auto rootType = cast<ttg::MemDescType>(channel.create.getBuffers()[0].getType());
    for (OpOperand &use : scope->token.getUses()) {
      Operation *user = use.getOwner();
      if (user == tokenEnd) {
        ++endUses;
        continue;
      }
      auto buffer = dyn_cast<ArefBufferOp>(user);
      if (!buffer || buffer.getAref() != channel.create.getResult() ||
          buffer.getStage() || buffer.getBuffers().size() != 1 ||
          !scope->contains(buffer))
        return false;
      auto type = cast<ttg::MemDescType>(buffer.getBuffers()[0].getType());
      if (type.getShape() != rootType.getShape().drop_front() ||
          type.getEncoding() != rootType.getEncoding() ||
          type.getMemorySpace() != rootType.getMemorySpace() ||
          type.getElementType() != rootType.getElementType() ||
          !validAllocationShape(type, rootType))
        return false;
      // Preserve the exact physical type on an enter->buffer borrow.
      if ((enter && type != buffers[0].getType()) ||
          (borrowedType && type != borrowedType))
        return false;
      buffers.push_back(buffer.getBuffers()[0]);
      recognized.insert(buffer);
    }
    if (endUses != 1)
      return false;
    DenseSet<Value> seen;
    for (Value buffer : buffers)
      if (!checkUses(buffer, *scope, seen))
        return false;
    if (!checkTokens(*scope))
      return false;
    for (Operation *access : scope->accesses)
      channel.mma |= isa<ttng::TCGen5MMAOp>(access);
    if (enter)
      recognized.insert(enter);
    if (scope->exit)
      recognized.insert(scope->exit);
    channel.scopes.push_back(std::move(scope));
    return true;
  }

  static bool checkUses(Value value, Scope &scope, DenseSet<Value> &seen) {
    if (!seen.insert(value).second)
      return true;
    for (Operation *user : value.getUsers()) {
      if (!scope.contains(user))
        return false;
      if (user->hasTrait<OpTrait::MemDescViewTrait>() &&
          user->getNumResults() == 1 &&
          isa<ttg::MemDescType>(user->getResult(0).getType())) {
        if (!checkUses(user->getResult(0), scope, seen))
          return false;
      } else if (auto load = dyn_cast<ttng::TMEMLoadOp>(user)) {
        if (load.getSrc() != value)
          return false;
        scope.accesses.insert(user);
      } else if (auto store = dyn_cast<ttng::TMEMStoreOp>(user)) {
        if (store.getDst() != value)
          return false;
        scope.accesses.insert(user);
      } else if (auto mma = dyn_cast<ttng::TCGen5MMAOp>(user)) {
        if ((mma.getA() != value && mma.getB() != value && mma.getD() != value) ||
            scope.completion != AsyncOp::TC5MMA || mma.getTwoCtas() ||
            !mma.getBarriers().empty())
          return false;
        // GET is an exclusive handoff too: accumulator ping-pong uses GET for
        // MMA read/write and PUT for correction. Issue is not completion; the
        // unchanged TC5 exit lowers to the commit protecting reuse.
        scope.accesses.insert(user);
      } else {
        return false;
      }
    }
    return true;
  }

  static std::pair<Value, Value> accessTokens(Operation *op) {
    if (auto load = dyn_cast<ttng::TMEMLoadOp>(op))
      return {load.getDep(), load.getToken()};
    if (auto store = dyn_cast<ttng::TMEMStoreOp>(op))
      return {store.getDep(), store.getToken()};
    auto mma = cast<ttng::TCGen5MMAOp>(op);
    return {mma.getAccDep(), mma.getToken()};
  }

  static bool checkTokens(const Scope &scope) {
    for (Operation *access : scope.accesses) {
      auto [dep, token] = accessTokens(access);
      if (dep) {
        Operation *def = dep.getDefiningOp();
        if (!def || !scope.accesses.contains(def) ||
            accessTokens(def).second != dep || !def->isBeforeInBlock(access))
          return false;
      }
      if (token)
        for (Operation *user : token.getUsers())
          if (!scope.accesses.contains(user) || accessTokens(user).first != token)
            return false;
    }
    return true;
  }

  DenseSet<Operation *> attempted;
  DenseMap<Operation *, std::unique_ptr<Channel>> channels;
};

struct ScheduleArefReleases
    : impl::NVWSScheduleArefReleasesBase<ScheduleArefReleases> {
  void runOnOperation() override {
    if (ttg::TritonGPUDialect::getNumCTAs(getOperation()) != 1)
      return;
    Ownership ownership;
    SmallVector<ArefGetExitOp> exits;
    getOperation().walk([&](ArefGetExitOp exit) { exits.push_back(exit); });
    for (ArefGetExitOp exit : exits) {
      ++numExamined;
      Scope *scope = ownership.getScope(exit);
      if (!scope || !ttg::hasPartition(scope->context))
        continue;
      ++numCertified;
      Operation *insertion = exit;
      unsigned crossedMemory = 0;
      while (Operation *previous = insertion->getPrevNode()) {
        if (!scope->contains(previous) || previous->getNumRegions() ||
            previous->hasTrait<OpTrait::IsTerminator>() ||
            isa<CallOpInterface>(previous) ||
            llvm::any_of(exit->getOperands(), [&](Value value) {
              return value.getDefiningOp() == previous;
            }))
          break;
        if (isMemoryEffectFree(previous) && isSpeculatable(previous)) {
          insertion = previous;
          continue;
        }
        if (!ownership.heldElsewhere(previous, exit, *scope))
          break;
        ++crossedMemory;
        insertion = previous;
      }
      if (insertion != exit.getOperation()) {
        exit->moveBefore(insertion);
        ++numMoved;
        numMemoryCrossed += crossedMemory;
      }
    }
    numChannels = ownership.size();
    numMultibuffer = ownership.multibufferCount();
    numCarried = ownership.carriedCount();
    numMma = ownership.mmaCount();
  }
};

} // namespace
} // namespace mlir::triton
