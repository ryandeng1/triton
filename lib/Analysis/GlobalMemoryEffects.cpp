#include "triton/Analysis/GlobalMemoryEffects.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
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
