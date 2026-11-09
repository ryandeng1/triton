#include "triton/Analysis/GlobalMemoryEffects.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::triton {

GlobalMemoryEffects::GlobalMemoryEffects(FuncOp func)
    : function(func.getOperation()) {
  hasUnknownEffects =
      func->walk([&](Operation *op) {
            if (op == function)
              return WalkResult::advance();
            if (auto load = dyn_cast<LoadOp>(op)) {
              if (load.getIsVolatile() || !getArgumentRoot(load.getPtr()))
                return WalkResult::interrupt();
              return WalkResult::advance();
            }
            if (auto store = dyn_cast<StoreOp>(op)) {
              auto arg = getArgumentRoot(store.getPtr());
              if (!arg)
                return WalkResult::interrupt();
              writtenArguments.push_back(arg);
              return WalkResult::advance();
            }
            if (isMemoryEffectFree(op) ||
                op->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
              return WalkResult::advance();
            return WalkResult::interrupt();
          })
          .wasInterrupted();
}

BlockArgument GlobalMemoryEffects::getArgumentRoot(Value pointer) const {
  if (!pointer || !isa<PointerType>(getElementTypeOrSelf(pointer.getType())))
    return {};
  auto findBase = [&](auto &&self, Value value, BlockArgument anchor,
                      unsigned budget) -> Value {
    if (!value || !budget)
      return {};
    while (Operation *def = value.getDefiningOp()) {
      if (!budget)
        return {};
      --budget;
      if (isa<AddPtrOp, SplatOp, BroadcastOp, ExpandDimsOp, ReshapeOp,
              gpu::ConvertLayoutOp>(def)) {
        value = def->getOperand(0);
        continue;
      }
      if (auto loop = dyn_cast<scf::ForOp>(def)) {
        auto result = cast<OpResult>(value);
        value = loop.getRegionIterArgs()[result.getResultNumber()];
        break;
      }
      // Casts, selections, calls and while results need separate proofs.
      return {};
    }
    auto arg = dyn_cast<BlockArgument>(value);
    if (!arg)
      return {};
    if (arg == anchor)
      return arg;
    if (auto func = dyn_cast<FuncOp>(arg.getOwner()->getParentOp()))
      return func.getOperation() == function &&
                     arg.getOwner() == &func.getBody().front()
                 ? Value(arg)
                 : Value();
    if (auto partitions =
            dyn_cast<gpu::WarpSpecializePartitionsOp>(
                arg.getOwner()->getParentOp())) {
      if (!budget || arg.getOwner() != &arg.getOwner()->getParent()->front() ||
          arg.getArgNumber() >= partitions.getExplicitCaptures().size())
        return {};
      return self(self, partitions.getExplicitCaptures()[arg.getArgNumber()],
                  anchor, budget - 1);
    }
    auto loop = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp());
    if (!loop || arg.getOwner() != loop.getBody() || arg.getArgNumber() == 0 ||
        !budget)
      return {};
    unsigned iter = arg.getArgNumber() - 1;
    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    // Each backedge must derive from this exact carried pointer. Following an
    // unrelated carry or merely trusting the initial pointer would be unsound.
    // Nested for results are accepted only after their own recurrence check.
    if (self(self, yield.getOperand(iter), arg, budget - 1) != arg)
      return {};
    return self(self, loop.getInitArgs()[iter], anchor, budget - 1);
  };
  Value root = findBase(findBase, pointer, BlockArgument(), 64);
  return root ? dyn_cast<BlockArgument>(root) : BlockArgument();
}

std::optional<int64_t> GlobalMemoryEffects::getAliasGroup(Value pointer) const {
  auto arg = getArgumentRoot(pointer);
  if (!arg)
    return std::nullopt;
  auto attr = cast<FuncOp>(function).getArgAttrOfType<IntegerAttr>(
      arg.getArgNumber(), "tt.experimental_alias_group");
  if (!attr || !attr.getType().isInteger(32) || attr.getInt() < 0)
    return std::nullopt;
  return attr.getInt();
}

bool GlobalMemoryEffects::isExplicitlyImmutable(Value pointer) const {
  auto arg = getArgumentRoot(pointer);
  if (!arg)
    return false;
  auto attr = cast<FuncOp>(function).getArgAttrOfType<IntegerAttr>(
      arg.getArgNumber(), "tt.experimental_immutable");
  return attr && attr.getType().isInteger(32) && attr.getInt() == 1;
}

bool GlobalMemoryEffects::isExplicitlyImmutableGlobal(Value pointer) const {
  auto arg = getArgumentRoot(pointer);
  if (!arg)
    return false;
  auto type = dyn_cast<PointerType>(arg.getType());
  return type && type.getAddressSpace() == PtrAddrSpace::Global &&
         isExplicitlyImmutable(arg);
}

bool GlobalMemoryEffects::mayAlias(Value first, Value second) const {
  auto firstGroup = getAliasGroup(first), secondGroup = getAliasGroup(second);
  return !firstGroup || !secondGroup || *firstGroup == *secondGroup;
}

bool GlobalMemoryEffects::mayModify(Value pointer) const {
  auto arg = getArgumentRoot(pointer);
  if (hasUnknownEffects || !arg)
    return true;
  return llvm::any_of(writtenArguments, [&](BlockArgument written) {
    return mayAlias(arg, written);
  });
}

namespace {
struct ScopedGlobalAccesses {
  SmallVector<Value> reads;
  SmallVector<Value> stores;
  bool supported = true;
};

bool isPrivateScratch(gpu::GlobalScratchAllocOp alloc) {
  return alloc && !alloc.getThirdPartyAllocation() &&
         !alloc.getSharedClusterState();
}

bool isPrivateDescriptorScratch(Value pointer) {
  auto type = dyn_cast<PointerType>(pointer.getType());
  if (!type || type.getAddressSpace() != PtrAddrSpace::Global)
    return false;
  // Descriptor-ring lowering uses scalar addptrs rooted in private scratch.
  // Do not look through casts, calls, region captures, or arbitrary pointers.
  for (unsigned budget = 64; budget; --budget) {
    if (auto alloc = pointer.getDefiningOp<gpu::GlobalScratchAllocOp>())
      return isPrivateScratch(alloc);
    auto add = pointer.getDefiningOp<AddPtrOp>();
    if (!add)
      return false;
    pointer = add.getPtr();
  }
  return false;
}

ScopedGlobalAccesses collectGlobalAccesses(Operation *scope,
                                         bool orderedDescriptorStores,
                                         bool allowCompilerProtocols = false) {
  ScopedGlobalAccesses accesses;
  accesses.supported = !scope->walk([&](Operation *op) {
    if (orderedDescriptorStores && op != scope && isa<LoopLikeOpInterface>(op))
      return WalkResult::interrupt();
    if (isa<CallOp, AtomicOpInterface, AtomicPollOp, gpu::BarrierOp>(op))
      return WalkResult::interrupt();
    if (auto copy = dyn_cast<gpu::AsyncCopyGlobalToLocalOp>(op);
        copy && copy.getIsVolatile())
      return WalkResult::interrupt();
    if (auto copy = dyn_cast<nvidia_gpu::AsyncTMACopyGlobalToLocalOp>(op);
        copy && copy.getIsVolatile())
      return WalkResult::interrupt();
    if (!allowCompilerProtocols &&
        isa<nvidia_gpu::InitBarrierOp, nvidia_gpu::InvalBarrierOp>(op))
      return WalkResult::interrupt();
    if (allowCompilerProtocols &&
        isa<gpu::AsyncWaitOp, gpu::AsyncCommitGroupOp,
             nvidia_gpu::FenceAsyncSharedOp,
             nvidia_gpu::FenceMBarrierInitReleaseClusterOp,
             nvidia_gpu::WarpGroupDotWaitOp, nvidia_gpu::TMAStoreWaitOp>(op))
      return WalkResult::advance();
    if (allowCompilerProtocols) {
      // Lowering tt.make_tensor_descriptor introduces private global scratch
      // and descriptor updates. Their unvalued GlobalMemory effects describe
      // the descriptor, not the user tensor whose address it encodes. Only the
      // compiler-proven original scope may rely on the descriptor ring's
      // lifetime/order protocol; source or arbitrary-address updates still
      // fail the generic global-write check below.
      if (auto alloc = dyn_cast<gpu::GlobalScratchAllocOp>(op);
          isPrivateScratch(alloc))
        return WalkResult::advance();
      if (auto create = dyn_cast<nvidia_gpu::TensormapCreateOp>(op);
          create && isPrivateDescriptorScratch(create.getDescPtr()))
        return WalkResult::advance();
      if (auto fence = dyn_cast<nvidia_gpu::TensormapFenceproxyAcquireOp>(op);
          fence && isPrivateDescriptorScratch(fence.getDescPtr()))
        return WalkResult::advance();
    }
    if (auto load = dyn_cast<LoadOp>(op)) {
      if (orderedDescriptorStores || load.getIsVolatile())
        return WalkResult::interrupt();
      accesses.reads.push_back(load.getPtr());
      return WalkResult::advance();
    }
    if (auto store = dyn_cast<StoreOp>(op)) {
      if (orderedDescriptorStores)
        return WalkResult::interrupt();
      accesses.stores.push_back(store.getPtr());
      return WalkResult::advance();
    }
    if (isa<DescriptorStoreLikeOpInterface>(op))
      return orderedDescriptorStores ? WalkResult::advance()
                                     : WalkResult::interrupt();
    if (auto effects = dyn_cast<MemoryEffectOpInterface>(op)) {
      SmallVector<MemoryEffects::EffectInstance> instances;
      effects.getEffects(instances);
      // Delaying a global store across a shared-memory release/barrier can
      // change publication even when the barrier has no global-memory effect.
      // The bounded store-only path admits ordinary local storage operations,
      // not arbitrary asynchronous shared/TMEM protocols.
      if (orderedDescriptorStores && !instances.empty() &&
          !isa<gpu::LocalAllocOp, gpu::LocalLoadOp, gpu::LocalStoreOp,
               gpu::LocalDeallocOp>(op))
        return WalkResult::interrupt();
      for (const auto &effect : instances) {
        if (!allowCompilerProtocols &&
            gpu::SharedMemoryEffect::classof(&effect) &&
            gpu::SharedMemoryEffect(&effect).getKind() ==
                gpu::SharedKind::Barrier)
          return WalkResult::interrupt();
        if (isa<gpu::SharedMemory, nvidia_gpu::TensorMemory>(
                effect.getResource()))
          continue;
        if (!orderedDescriptorStores &&
            isa<GlobalMemory>(effect.getResource()) &&
            isa<MemoryEffects::Read>(effect.getEffect())) {
          // A descriptor cannot currently resolve to an argument pointer. It
          // therefore conflicts conservatively with any ordinary store.
          accesses.reads.push_back(effect.getValue());
          continue;
        }
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    }
    return isMemoryEffectFree(op) ||
                   op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()
               ? WalkResult::advance()
               : WalkResult::interrupt();
  }).wasInterrupted();
  return accesses;
}
} // namespace

bool GlobalMemoryEffects::hasIndependentGlobalMemoryAccesses(
    Operation *scope, bool allowCompilerProtocols) const {
  auto accesses = collectGlobalAccesses(scope, false, allowCompilerProtocols);
  if (!accesses.supported)
    return false;
  for (Value read : accesses.reads) {
    if (isExplicitlyImmutableGlobal(read))
      continue;
    if (llvm::any_of(accesses.stores,
                     [&](Value store) { return mayAlias(read, store); }))
      return false;
  }
  for (auto [index, store] : llvm::enumerate(accesses.stores))
    for (Value other : ArrayRef<Value>(accesses.stores).drop_front(index + 1))
      if (mayAlias(store, other))
        return false;
  return true;
}

bool GlobalMemoryEffects::hasOnlyOrderedDescriptorStores(Operation *scope) const {
  return collectGlobalAccesses(scope, true).supported;
}

bool GlobalMemoryEffects::hasOnlyOrdinaryEffects(
    Operation *scope, bool allowImmutableCTABarriers) {
  return !scope
              ->walk([scope, allowImmutableCTABarriers](Operation *op) {
                if (op == scope && isa<FuncOp>(op))
                  return WalkResult::advance();
                if (auto barrier = dyn_cast<gpu::BarrierOp>(op))
                  return allowImmutableCTABarriers &&
                                 barrier.getScope() == gpu::BarrierScope::CTA
                             ? WalkResult::advance()
                             : WalkResult::interrupt();
                if (auto load = dyn_cast<LoadOp>(op))
                  return load.getIsVolatile() ? WalkResult::interrupt()
                                              : WalkResult::advance();
                if (isa<StoreOp>(op) || isMemoryEffectFree(op) ||
                    op->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
                  return WalkResult::advance();
                return WalkResult::interrupt();
              })
              .wasInterrupted();
}

} // namespace mlir::triton
