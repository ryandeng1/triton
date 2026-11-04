#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir::triton::nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUPACKTMEMPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {
namespace ttg = gpu;

static bool constant(Value value, int64_t expected, bool absent = false) {
  if (!value)
    return absent;
  APInt result;
  return matchPattern(value, m_ConstantInt(&result)) &&
         result.getSExtValue() == expected;
}

// Distinguish issue from completion. Program order connects issue nodes; an
// asynchronous MMA completion reaches another partition only through its
// commit's completion notification and the matching barrier wait.
// Ordinary TMEM loads have synchronous block-level semantics. Their completion
// before publication is implemented later by TMemBarrierInsertion's waits and
// warp rendezvous; this graph relies on that existing lowering obligation.
struct EventGraph {
  SmallVector<SmallVector<unsigned>> edges;
  DenseMap<Operation *, unsigned> issue, complete;

  unsigned node() {
    edges.emplace_back();
    return edges.size() - 1;
  }
  void edge(unsigned from, unsigned to) { edges[from].push_back(to); }
  bool reaches(unsigned from, unsigned to) const {
    DenseSet<unsigned> seen;
    SmallVector<unsigned> pending{from};
    while (!pending.empty()) {
      unsigned current = pending.pop_back_val();
      if (!seen.insert(current).second)
        continue;
      for (unsigned next : edges[current]) {
        if (next == to)
          return true;
        pending.push_back(next);
      }
    }
    return false;
  }
  bool acyclic() const {
    for (unsigned i = 0; i < edges.size(); ++i)
      if (reaches(i, i))
        return false;
    return true;
  }
};

struct Access {
  Operation *op;
  int64_t begin, end;
};

struct Allocation {
  TMEMAllocOp op;
  SmallVector<Access> accesses;
  SmallVector<BlockArgument> captures;
  TCGen5MMAOp writer;
};

struct Barrier {
  ttg::LocalAllocOp alloc;
  InitBarrierOp init;
  Operation *signal = nullptr;
  WaitBarrierOp wait;
};

// This first recognizer is deliberately bounded: two straight-line partitions,
// one generation per freshly initialized barrier, and full-row f32 TMEM. It
// preserves the existing protocol rather than synthesizing partial releases.
class PackingProof {
public:
  explicit PackingProof(ttg::WarpSpecializeOp ws) : ws(ws) {}

  bool analyze() {
    if (!isa<triton::FuncOp>(ws->getParentOp()) ||
        !llvm::hasSingleElement(ws.getDefaultRegion()) ||
        ws.getPartitionRegions().size() != 1)
      return false;
    blocks.push_back(&ws.getDefaultRegion().front());
    for (Region *region : ws.getPartitionRegions()) {
      if (!llvm::hasSingleElement(*region))
        return false;
      blocks.push_back(&region->front());
    }
    for (Block *block : blocks) {
      SmallVector<unsigned> pendingMmas;
      std::optional<unsigned> previous, previousCommit;
      for (Operation &op : *block) {
        if (op.getNumRegions() || !allowed(&op))
          return false;
        unsigned issue = graph.node();
        graph.issue[&op] = issue;
        graph.complete[&op] = issue;
        if (previous)
          graph.edge(*previous, issue);
        previous = issue;
        if (auto mma = dyn_cast<TCGen5MMAOp>(op)) {
          if (!validMma(mma))
            return false;
          unsigned done = graph.node();
          graph.complete[&op] = done;
          graph.edge(issue, done);
          pendingMmas.push_back(done);
        } else if (auto commit = dyn_cast<TCGen5CommitOp>(op)) {
          if (!constant(commit.getPred(), -1, true) ||
              !commit.getDescs().empty())
            return false;
          unsigned done = graph.node();
          graph.complete[&op] = done;
          graph.edge(issue, done);
          for (unsigned mma : pendingMmas)
            graph.edge(mma, done);
          pendingMmas.clear();
          if (previousCommit)
            graph.edge(*previousCommit, done);
          previousCommit = done;
        }
        Value barrier;
        if (auto wait = dyn_cast<WaitBarrierOp>(op))
          barrier = wait.getAlloc();
        else if (auto arrive = dyn_cast<ArriveBarrierOp>(op))
          barrier = arrive.getAlloc();
        else if (auto commit = dyn_cast<TCGen5CommitOp>(op))
          barrier = commit.getBarrier();
        if (barrier && !certifyBarrier(barrier))
          return false;
      }
      if (!pendingMmas.empty())
        return false;
    }
    for (auto &[root, barrier] : barriers) {
      if (!barrier.signal || !barrier.wait ||
          barrier.signal->getBlock() == barrier.wait->getBlock())
        return false;
      graph.edge(graph.complete.lookup(barrier.signal),
                 graph.issue.lookup(barrier.wait));
    }
    return graph.acyclic();
  }

  std::optional<Allocation> allocation(TMEMAllocOp alloc, bool allowSlices) {
    auto type = alloc.getType();
    auto encoding = dyn_cast<TensorMemoryEncodingAttr>(type.getEncoding());
    if (alloc->getBlock() != ws->getBlock() || !alloc->isBeforeInBlock(ws) ||
        alloc.getSrc() || (alloc.getToken() && !alloc.getToken().use_empty()) ||
        alloc->hasAttr("tensor_memory_col_offset") ||
        alloc->hasAttr("tensor_memory_row_offset") || type.getRank() != 2 ||
        type.getShape()[0] != 128 || !type.getElementType().isF32() ||
        type.getAllocShape() != type.getShape() || !encoding ||
        encoding.getBlockM() != 128 || encoding.getColStride() != 1 ||
        encoding.getTwoCTAs() ||
        (encoding.getBlockN() != 64 && encoding.getBlockN() != 128) ||
        (type.getShape()[1] != 64 && type.getShape()[1] != 128 &&
         type.getShape()[1] != 256) ||
        type.getShape()[1] % encoding.getBlockN())
      return std::nullopt;
    auto physical = getTmemAllocSizes(type);
    if (physical.numRows != 128 || physical.numCols != type.getShape()[1])
      return std::nullopt;
    Allocation result{alloc, {}, {}, {}};
    DenseSet<Value> seen;
    if (!collect(alloc.getResult(), 0, type.getShape()[1], allowSlices,
                 result, seen) || !result.writer || result.accesses.size() < 2)
      return std::nullopt;
    // A single complete overwrite initializes each logical allocation. Every
    // read must observe that MMA's completion, never just its issue token.
    for (const Access &access : result.accesses)
      if (access.op != result.writer.getOperation() &&
          !graph.reaches(graph.complete.lookup(result.writer),
                         graph.issue.lookup(access.op)))
        return std::nullopt;
    return result;
  }

  bool fits(const Allocation &a, const Allocation &b, int64_t offset) const {
    // Pairwise race freedom alone is insufficient: A.write, B.write, A.read is
    // ordered but B would destroy a live A value. Require the *entire* A value
    // lifetime in the overlap to finish before every B access starts. Include
    // read/read pairs, too, so the direction cannot vary between accesses.
    for (const Access &first : a.accesses)
      for (const Access &second : b.accesses) {
        if (first.end <= second.begin + offset ||
            second.end + offset <= first.begin)
          continue;
        if (!graph.reaches(graph.complete.lookup(first.op),
                           graph.issue.lookup(second.op)))
          return false;
      }
    return true;
  }

  unsigned completionEdges() const { return barriers.size(); }

  bool completionEffectsAvailable(const Allocation &allocation) const {
    for (const auto &[root, barrier] : barriers)
      if (isa<TCGen5CommitOp>(barrier.signal) &&
          graph.reaches(graph.complete.lookup(allocation.writer),
                        graph.complete.lookup(barrier.signal)) &&
          barrier.wait->getBlock() != blocks.front() &&
          !llvm::any_of(allocation.captures, [&](BlockArgument capture) {
            return capture.getOwner() == barrier.wait->getBlock();
          }))
        return false;
    return true;
  }

  void retainCompletionEffects(const Allocation &allocation, Value storage) {
    for (auto &[root, barrier] : barriers) {
      if (!isa<TCGen5CommitOp>(barrier.signal) ||
          !graph.reaches(graph.complete.lookup(allocation.writer),
                         graph.complete.lookup(barrier.signal)))
        continue;
      Value dependency = storage;
      if (barrier.wait->getBlock() != &ws.getDefaultRegion().front()) {
        dependency = {};
        for (BlockArgument capture : allocation.captures)
          if (capture.getOwner() == barrier.wait->getBlock()) {
            dependency = capture;
            break;
          }
      }
      // Each recognized writer is a direct captured allocation, and the wait
      // is in the other partition. Retain its asynchronous memory effects at
      // the wait so later CSE cannot merge loads from different generations
      // after the two formerly independent allocations become aliases.
      if (dependency &&
          !llvm::is_contained(barrier.wait.getDeps(), dependency))
        barrier.wait.getDepsMutable().append(dependency);
    }
  }

private:
  bool inPartition(Operation *op) const {
    return llvm::is_contained(blocks, op->getBlock());
  }
  static bool validMma(TCGen5MMAOp mma) {
    return mma.getIsAsync() && !mma.getTwoCtas() && !mma.getMulticast() &&
           !mma.getAccDep() && (!mma.getToken() || mma.getToken().use_empty()) &&
           constant(mma.getPred(), -1) && constant(mma.getUseD(), 0) &&
           mma.getBarriers().empty() &&
           isa<ttg::SharedMemorySpaceAttr>(mma.getA().getType().getMemorySpace()) &&
           isa<ttg::SharedMemorySpaceAttr>(mma.getB().getType().getMemorySpace());
  }
  static bool allowed(Operation *op) {
    return isMemoryEffectFree(op) ||
           isa<TMEMLoadOp, TCGen5MMAOp, TCGen5CommitOp, WaitBarrierOp,
               ArriveBarrierOp, triton::LoadOp, triton::StoreOp>(op);
  }
  Value origin(Value value) const {
    while (auto arg = dyn_cast<BlockArgument>(value)) {
      auto partition = dyn_cast<ttg::WarpSpecializePartitionsOp>(
          arg.getOwner()->getParentOp());
      if (!partition || partition->getParentOp() != ws ||
          !llvm::is_contained(blocks, arg.getOwner()))
        return {};
      value = partition.getExplicitCaptures()[arg.getArgNumber()];
    }
    return value;
  }

  bool certifyBarrier(Value descriptor) {
    Value root = origin(descriptor);
    auto alloc = root ? root.getDefiningOp<ttg::LocalAllocOp>() : nullptr;
    if (!alloc || alloc.getSrc() || alloc->hasAttr("allocation.offset") ||
        alloc->getBlock() != ws->getBlock() ||
        !alloc->isBeforeInBlock(ws) || alloc.getType().getShape() != ArrayRef<int64_t>{1} ||
        !alloc.getType().getElementType().isInteger(64))
      return false;
    if (barriers.contains(root))
      return true;
    Barrier barrier{alloc, {}, nullptr, {}};
    DenseSet<Value> seen;
    if (!barrierUses(root, barrier, seen) || !barrier.init || !barrier.signal ||
        !barrier.wait)
      return false;
    barriers[root] = barrier;
    return true;
  }
  bool barrierUses(Value value, Barrier &barrier, DenseSet<Value> &seen) {
    if (!seen.insert(value).second)
      return true;
    for (OpOperand &use : value.getUses()) {
      Operation *op = use.getOwner();
      if (auto partition = dyn_cast<ttg::WarpSpecializePartitionsOp>(op)) {
        if (partition->getParentOp() != ws)
          return false;
        for (Region &region : partition.getPartitionRegions())
          if (!barrierUses(region.front().getArgument(use.getOperandNumber()),
                           barrier, seen))
            return false;
      } else if (auto init = dyn_cast<InitBarrierOp>(op)) {
        if (barrier.init || init.getCount() != 1 ||
            op->getBlock() != ws->getBlock() || !op->isBeforeInBlock(ws))
          return false;
        barrier.init = init;
      } else if (auto inval = dyn_cast<InvalBarrierOp>(op)) {
        if (op->getBlock() != ws->getBlock() || !ws->isBeforeInBlock(op))
          return false;
      } else if (auto wait = dyn_cast<WaitBarrierOp>(op)) {
        if (barrier.wait || !inPartition(op) ||
            !constant(wait.getPhase(), 0) ||
            !constant(wait.getPred(), -1, true) || !wait.getDeps().empty())
          return false;
        barrier.wait = wait;
      } else if (auto arrive = dyn_cast<ArriveBarrierOp>(op)) {
        if (barrier.signal || !inPartition(op) || arrive.getCount() != 1 ||
            !constant(arrive.getPred(), -1, true) || arrive.getFromCTA() ||
            arrive.getMulticastCTA() || arrive.getPerWarp() ||
            !arrive.getReleaseResources().empty())
          return false;
        barrier.signal = op;
      } else if (auto commit = dyn_cast<TCGen5CommitOp>(op)) {
        if (barrier.signal || !inPartition(op) ||
            !constant(commit.getPred(), -1, true) ||
            !commit.getDescs().empty())
          return false;
        barrier.signal = op;
      } else {
        return false;
      }
    }
    return true;
  }

  bool collect(Value value, int64_t offset, int64_t width, bool allowSlices,
               Allocation &allocation, DenseSet<Value> &seen) {
    if (!seen.insert(value).second)
      return true;
    for (OpOperand &use : value.getUses()) {
      Operation *op = use.getOwner();
      if (auto partition = dyn_cast<ttg::WarpSpecializePartitionsOp>(op)) {
        if (partition->getParentOp() != ws)
          return false;
        for (Region &region : partition.getPartitionRegions()) {
          BlockArgument arg = region.front().getArgument(use.getOperandNumber());
          if (arg.getType() != value.getType())
            return false;
          allocation.captures.push_back(arg);
          if (!collect(arg, offset, width, allowSlices, allocation, seen))
            return false;
        }
      } else if (auto slice = dyn_cast<TMEMSubSliceOp>(op)) {
        auto encoding = cast<TensorMemoryEncodingAttr>(
            allocation.op.getType().getEncoding());
        if (!allowSlices || !inPartition(op) || slice.getDim() != 1 ||
            slice.getOffset() < 0 ||
            slice.getOffset() + slice.getType().getShape()[1] > width ||
            slice.getOffset() % encoding.getBlockN() ||
            slice.getType().getShape()[1] % encoding.getBlockN() ||
            getTMemSubSliceOffset(slice.getSrc().getType(), slice.getOffset(),
                                 1) != slice.getOffset())
          return false;
        if (!collect(slice.getResult(), offset + slice.getOffset(),
                     slice.getType().getShape()[1], allowSlices, allocation, seen))
          return false;
      } else if (auto load = dyn_cast<TMEMLoadOp>(op)) {
        if (!inPartition(op) || load.getSrc() != value || load.getDep() ||
            (load.getToken() && !load.getToken().use_empty()))
          return false;
        allocation.accesses.push_back({op, offset, offset + width});
      } else if (auto mma = dyn_cast<TCGen5MMAOp>(op)) {
        if (!inPartition(op) || !validMma(mma) || mma.getD() != value ||
            allocation.writer || offset != 0 ||
            width != allocation.op.getType().getShape()[1])
          return false;
        allocation.writer = mma;
        allocation.accesses.push_back({op, offset, offset + width});
      } else {
        return false;
      }
    }
    return true;
  }

  ttg::WarpSpecializeOp ws;
  SmallVector<Block *> blocks;
  DenseMap<Value, Barrier> barriers;
  EventGraph graph;
};

struct PackTMem : impl::TritonNvidiaGPUPackTMemPassBase<PackTMem> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (ttg::TritonGPUDialect::getNumCTAs(module) != 1)
      return;
    module.walk([&](ttg::WarpSpecializeOp ws) {
      PackingProof proof(ws);
      if (!proof.analyze())
        return;
      SmallVector<TMEMAllocOp> allocs;
      for (Operation &op : *ws->getBlock())
        if (auto alloc = dyn_cast<TMEMAllocOp>(op))
          allocs.push_back(alloc);
      for (TMEMAllocOp first : allocs) {
        auto a = proof.allocation(first, true);
        if (!a)
          continue;
        for (TMEMAllocOp second : allocs) {
          if (first == second ||
              first.getType().getEncoding() != second.getType().getEncoding())
            continue;
          auto b = proof.allocation(second, false);
          if (!b || !proof.completionEffectsAvailable(*a) ||
              !proof.completionEffectsAvailable(*b))
            continue;
          ++numCandidates;
          int64_t aWidth = first.getType().getShape()[1];
          int64_t bWidth = second.getType().getShape()[1];
          int alignment = cast<TensorMemoryEncodingAttr>(
                              first.getType().getEncoding()).getBlockN();
          for (int64_t offset = 0; offset + bWidth <= aWidth;
               offset += alignment) {
            if (!proof.fits(*a, *b, offset))
              continue;
            OpBuilder builder(ws);
            auto slice = TMEMSubSliceOp::create(builder, second.getLoc(),
                                               first.getResult(), offset,
                                               bWidth, 1);
            for (BlockArgument arg : b->captures)
              arg.setType(slice.getType());
            second.getResult().replaceAllUsesWith(slice.getResult());
            proof.retainCompletionEffects(*a, first.getResult());
            proof.retainCompletionEffects(*b, slice.getResult());
            second.erase();
            ++numPacked;
            numColumns += bWidth;
            numWaitEdges += proof.completionEdges();
            // Keep the graph immutable: one checked placement per WS region.
            return;
          }
        }
      }
    });
  }
};

} // namespace
} // namespace mlir::triton::nvidia_gpu
