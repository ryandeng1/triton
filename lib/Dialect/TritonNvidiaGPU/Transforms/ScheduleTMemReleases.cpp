#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Analysis/BufferRegion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/ScopeExit.h"

#include <functional>
#include <memory>

namespace mlir::triton::nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUSCHEDULETMEMRELEASESPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

// Equalities have a canonical unordered operand pair. A null second operand
// denotes an otherwise opaque Boolean SSA value. Keeping conditions attached
// to views prevents a pair of correlated scf.if operations from inventing
// impossible combinations of accumulator base and subtile offset.
using Predicate = std::pair<Value, Value>;
using Conditions = DenseMap<Predicate, bool>;

static bool addCondition(Conditions &conditions, Value condition, bool truth) {
  APInt constant;
  if (matchPattern(condition, m_ConstantInt(&constant)))
    return constant.getBoolValue() == truth;

  if (auto xorOp = condition.getDefiningOp<arith::XOrIOp>()) {
    if (matchPattern(xorOp.getRhs(), m_ConstantInt(&constant)) &&
        constant.getBitWidth() == 1)
      return addCondition(conditions, xorOp.getLhs(),
                          truth != constant.getBoolValue());
  }

  Predicate key{condition, Value()};
  if (auto cmp = condition.getDefiningOp<arith::CmpIOp>()) {
    auto pred = cmp.getPredicate();
    if (pred == arith::CmpIPredicate::eq ||
        pred == arith::CmpIPredicate::ne) {
      key = {cmp.getLhs(), cmp.getRhs()};
      if (std::less<const void *>()(key.second.getAsOpaquePointer(),
                                    key.first.getAsOpaquePointer()))
        std::swap(key.first, key.second);
      if (pred == arith::CmpIPredicate::ne)
        truth = !truth;
    }
  }
  auto [it, inserted] = conditions.try_emplace(key, truth);
  return inserted || it->second == truth;
}

static bool compatible(const Conditions &lhs, const Conditions &rhs) {
  for (const auto &[key, truth] : lhs) {
    auto it = rhs.find(key);
    if (it != rhs.end() && it->second != truth)
      return false;
  }
  return true;
}

struct ConditionalView {
  BufferRegionView view;
  Conditions conditions;
};
using Views = SmallVector<std::shared_ptr<const ConditionalView>>;

// Use BufferRegion's physical layout/padding/CTA geometry without requiring
// materialized allocation offsets or running its control-flow dataflow. Fresh
// TMEM allocation objects provide logical provenance before allocation. When
// offsets already exist, compare their actual module-wide physical addresses.
class DescriptorRegions {
public:
  DescriptorRegions() {
    geometry = solver.load<BufferRegionAnalysis>(
        BufferRegionAnalysis::Mode::TensorMemoryOnly);
  }

  FailureOr<Views> resolve(Value value) {
    DenseSet<Value> active;
    return resolve(value, {}, active);
  }

  static bool mayOverlap(const Views &lhs, const Views &rhs) {
    for (const auto &a : lhs) {
      for (const auto &b : rhs) {
        if (!compatible(a->conditions, b->conditions))
          continue;
        Operation *aRoot = a->view.allocation;
        Operation *bRoot = b->view.allocation;
        if (aRoot != bRoot) {
          bool aAssigned = aRoot->hasAttr("tensor_memory_col_offset");
          bool bAssigned = bRoot->hasAttr("tensor_memory_col_offset");
          // Distinct fresh allocation objects are logically disjoint. The
          // subsequent allocator must preserve simultaneous logical lifetimes.
          if (!aAssigned && !bAssigned)
            continue;
          if (aAssigned != bAssigned)
            return true;
        }
        if (a->view.region.intersects(b->view.region))
          return true;
      }
    }
    return false;
  }

private:
  static constexpr unsigned maxVariants = 64;
  DataFlowSolver solver;
  BufferRegionAnalysis *geometry;

  static void append(Views &dest, const Views &source) {
    for (const auto &view : source)
      dest.push_back(view);
  }

  FailureOr<Views> resolve(Value value, const Conditions &conditions,
                           DenseSet<Value> &active) {
    auto type = dyn_cast<gpu::MemDescType>(value.getType());
    if (!type || !isa<TensorMemorySpaceAttr>(type.getMemorySpace()) ||
        active.size() >= 64 || !active.insert(value).second)
      return failure();
    auto guard = llvm::scope_exit([&] { active.erase(value); });

    if (auto arg = dyn_cast<BlockArgument>(value)) {
      Operation *parent = arg.getOwner()->getParentOp();
      if (auto partitions = dyn_cast<gpu::WarpSpecializePartitionsOp>(parent))
        return resolve(partitions.getExplicitCaptures()[arg.getArgNumber()],
                       conditions, active);
      // A trivially forwarded for-loop argument is invariant. Other loop/CFG
      // recurrences remain unknown; never correlate different iterations.
      if (auto loop = dyn_cast<scf::ForOp>(parent)) {
        unsigned index = arg.getArgNumber();
        if (index && loop.getYieldedValues()[index - 1] == value)
          return resolve(loop.getInitArgs()[index - 1], conditions, active);
      }
      return failure();
    }

    if (auto alloc = value.getDefiningOp<TMEMAllocOp>()) {
      auto col = alloc->getAttrOfType<IntegerAttr>("tensor_memory_col_offset");
      auto row = alloc->getAttrOfType<IntegerAttr>("tensor_memory_row_offset");
      if (bool(col) != bool(row))
        return failure();
      uint32_t base = col ? (uint32_t(col.getInt()) |
                              (uint32_t(row.getInt()) << 16))
                          : 0;
      BufferRegionView seed({}, base, 0, {}, 0, 0, 0, alloc);
      auto view = geometry->getSubView(type, seed);
      return Views{std::make_shared<ConditionalView>(
          ConditionalView{std::move(view), conditions})};
    }

    if (auto ifOp = value.getDefiningOp<scf::IfOp>()) {
      if (ifOp.getElseRegion().empty())
        return failure();
      unsigned result = cast<OpResult>(value).getResultNumber();
      Views views;
      for (bool thenBranch : {true, false}) {
        Conditions path = conditions;
        if (!addCondition(path, ifOp.getCondition(), thenBranch))
          continue;
        auto yield = thenBranch ? ifOp.thenYield() : ifOp.elseYield();
        auto branch = resolve(yield.getOperand(result), path, active);
        if (failed(branch))
          return failure();
        append(views, *branch);
        if (views.size() > maxVariants)
          return failure();
      }
      return views;
    }

    if (auto select = value.getDefiningOp<arith::SelectOp>()) {
      Views views;
      for (bool trueValue : {true, false}) {
        Conditions path = conditions;
        if (!addCondition(path, select.getCondition(), trueValue))
          continue;
        auto branch = resolve(trueValue ? select.getTrueValue()
                                        : select.getFalseValue(),
                              path, active);
        if (failed(branch))
          return failure();
        append(views, *branch);
        if (views.size() > maxVariants)
          return failure();
      }
      return views;
    }

    auto subview = [&](Value source, ArrayRef<uint32_t> offsets,
                       bool storageOffset = false) -> FailureOr<Views> {
      auto sources = resolve(source, conditions, active);
      if (failed(sources))
        return failure();
      Views views;
      for (const auto &sourceView : *sources) {
        for (uint32_t offset : offsets) {
          auto view = geometry->getSubView(
              type, sourceView->view, storageOffset ? offset : 0,
              storageOffset ? 0 : offset);
          views.push_back(std::make_shared<ConditionalView>(
              ConditionalView{std::move(view), sourceView->conditions}));
          if (views.size() > maxVariants)
            return failure();
        }
      }
      return views;
    };

    if (auto slice = value.getDefiningOp<TMEMSubSliceOp>()) {
      uint32_t offset = getTMemSubSliceOffset(
          slice.getSrc().getType(), slice.getOffset(), slice.getDim());
      return subview(slice.getSrc(), {offset});
    }
    if (auto index = value.getDefiningOp<gpu::MemDescIndexOp>()) {
      auto sourceType = index.getSrc().getType();
      if (sourceType.getRank() != 3)
        return failure();
      int64_t count = sourceType.getShape()[0];
      APInt constant;
      SmallVector<uint32_t> offsets;
      if (matchPattern(index.getIndex(), m_ConstantInt(&constant))) {
        int64_t i = constant.getSExtValue();
        if (i < 0 || i >= count)
          return failure();
        offsets.push_back(getTMemSubSliceOffset(sourceType, i, 0));
      } else {
        if (count > maxVariants)
          return failure();
        for (int64_t i = 0; i < count; ++i)
          offsets.push_back(getTMemSubSliceOffset(sourceType, i, 0));
      }
      return subview(index.getSrc(), offsets, /*storageOffset=*/true);
    }
    if (auto reinterpret = value.getDefiningOp<gpu::MemDescReinterpretOp>())
      return subview(reinterpret.getSrc(), {0});
    if (isa_and_nonnull<gpu::MemDescTransOp, gpu::MemDescReshapeOp>(
            value.getDefiningOp()))
      return resolve(value.getDefiningOp()->getOperand(0), conditions, active);
    return failure();
  }
};

static bool definesArrivalOperand(Operation *candidate, ArriveBarrierOp arrive) {
  return llvm::any_of(arrive->getOperands(), [&](Value operand) {
    Operation *definition = operand.getDefiningOp();
    return definition && (definition == candidate ||
                          candidate->isProperAncestor(definition));
  });
}

static bool isMovablePure(Operation *op) {
  if (isa<CallOpInterface>(op))
    return false;
  if (op->getNumRegions()) {
    // Even a memory-effect-free loop can diverge. Only view/arithmetic
    // conditionals are allowed; never cross iteration or execution regions.
    if (!isa<scf::IfOp>(op))
      return false;
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (Operation &nested : block) {
          if (isa<scf::YieldOp>(nested))
            continue;
          if (!isMovablePure(&nested))
            return false;
        }
    return true;
  }
  return isMemoryEffectFree(op) && isSpeculatable(op);
}

// Publication also writes the mbarrier itself. TMEM exclusivity does not allow
// moving that write across an aliased shared payload access. This small origin
// proof is deliberately restricted to fresh shared allocations; materialized
// physical offsets and unknown/escaping origins require a different analysis.
static bool collectSharedRoots(Value value, SmallVectorImpl<Operation *> &roots,
                               DenseSet<Value> &active) {
  auto type = dyn_cast<gpu::MemDescType>(value.getType());
  if (!type || !isa<gpu::SharedMemorySpaceAttr>(type.getMemorySpace()) ||
      active.size() >= 64 || !active.insert(value).second)
    return false;
  auto guard = llvm::scope_exit([&] { active.erase(value); });
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    if (auto partitions = dyn_cast<gpu::WarpSpecializePartitionsOp>(
            arg.getOwner()->getParentOp()))
      return collectSharedRoots(
          partitions.getExplicitCaptures()[arg.getArgNumber()], roots, active);
    return false;
  }
  Operation *op = value.getDefiningOp();
  if (isa<gpu::LocalAllocOp>(op)) {
    if (op->hasAttr("allocation.offset"))
      return false;
    roots.push_back(op);
    return true;
  }
  if (op->hasTrait<OpTrait::MemDescViewTrait>())
    return collectSharedRoots(op->getOperand(0), roots, active);
  if (auto select = dyn_cast<arith::SelectOp>(op))
    return collectSharedRoots(select.getTrueValue(), roots, active) &&
           collectSharedRoots(select.getFalseValue(), roots, active);
  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    if (ifOp.getElseRegion().empty())
      return false;
    unsigned index = cast<OpResult>(value).getResultNumber();
    return collectSharedRoots(ifOp.thenYield().getOperand(index), roots,
                              active) &&
           collectSharedRoots(ifOp.elseYield().getOperand(index), roots, active);
  }
  return false;
}

static bool sharedPayloadDisjointFromBarrier(Value payload,
                                             ArriveBarrierOp arrive) {
  SmallVector<Operation *> payloadRoots, barrierRoots;
  DenseSet<Value> active;
  if (!collectSharedRoots(payload, payloadRoots, active) ||
      !collectSharedRoots(arrive.getAlloc(), barrierRoots, active))
    return false;
  return llvm::none_of(payloadRoots, [&](Operation *root) {
    return llvm::is_contained(barrierRoots, root);
  });
}

static bool outputWaitDisjointFromBarrier(ArriveBarrierOp arrive) {
  auto function = arrive->getParentOfType<triton::FuncOp>();
  if (!function)
    return false;
  auto result = function.walk([&](Operation *op) {
    if (isa<CallOpInterface>(op))
      return WalkResult::interrupt();
    if (auto store = dyn_cast<TMAStoreLikeOpInterface>(op))
      if (!sharedPayloadDisjointFromBarrier(store.getSrc(), arrive))
        return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return !result.wasInterrupted();
}

// Crossing a wait/arrive/commit, opaque effect, or side-effecting region would
// need a protocol proof beyond an exclusive-resource declaration. The output
// whitelist is deliberately narrow: these operations neither access TMEM nor
// notify a peer through an mbarrier. Their relative order is never changed.
static bool canCross(Operation *op, ArriveBarrierOp arrive,
                     const Views &protectedViews, DescriptorRegions &regions) {
  if (definesArrivalOperand(op, arrive) ||
      op->hasTrait<OpTrait::IsTerminator>())
    return false;
  if (isMovablePure(op))
    return true;
  if (auto store = dyn_cast<gpu::LocalStoreOp>(op))
    return sharedPayloadDisjointFromBarrier(store.getDst(), arrive);
  if (auto store = dyn_cast<AsyncTMACopyLocalToGlobalOp>(op))
    return sharedPayloadDisjointFromBarrier(store.getSrc(), arrive);
  if (isa<TMAStoreWaitOp>(op))
    return outputWaitDisjointFromBarrier(arrive);
  Value accessed;
  if (auto load = dyn_cast<TMEMLoadOp>(op))
    accessed = load.getSrc();
  else if (auto store = dyn_cast<TMEMStoreOp>(op))
    accessed = store.getDst();
  else
    return false;
  auto accessViews = regions.resolve(accessed);
  return succeeded(accessViews) &&
         !DescriptorRegions::mayOverlap(protectedViews, *accessViews);
}

struct ScheduleTMemReleases
    : impl::TritonNvidiaGPUScheduleTMemReleasesPassBase<ScheduleTMemReleases> {
  void runOnOperation() override {
    DescriptorRegions regions;
    SmallVector<ArriveBarrierOp> arrivals;
    getOperation().walk([&](ArriveBarrierOp op) {
      if (!op.getReleaseResources().empty())
        arrivals.push_back(op);
    });
    for (ArriveBarrierOp arrive : arrivals) {
      Views protectedViews;
      bool known = true;
      for (Value resource : arrive.getReleaseResources()) {
        auto views = regions.resolve(resource);
        if (failed(views)) {
          known = false;
          break;
        }
        for (const auto &view : *views)
          protectedViews.push_back(view);
      }
      if (!known || protectedViews.empty())
        continue;
      Operation *insertion = arrive;
      while (Operation *previous = insertion->getPrevNode()) {
        if (!canCross(previous, arrive, protectedViews, regions))
          break;
        insertion = previous;
      }
      if (insertion != arrive.getOperation())
        arrive->moveBefore(insertion);
    }
  }
};

} // namespace
} // namespace mlir::triton::nvidia_gpu
