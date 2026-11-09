#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "triton/Analysis/GlobalMemoryEffects.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"

namespace mlir::triton {

#define GEN_PASS_DEF_TRITONLOOPINVARIANTCODEMOTION
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

#define DEBUG_TYPE "triton-licm"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

class LoopInvariantCodeMotionPass
    : public impl::TritonLoopInvariantCodeMotionBase<
          LoopInvariantCodeMotionPass> {
public:
  using TritonLoopInvariantCodeMotionBase::TritonLoopInvariantCodeMotionBase;

private:
  DenseMap<LoopLikeOpInterface, bool> isLoopMemoryEffectFreeOrOnlyRead;
  DenseMap<Operation *, GlobalMemoryEffects> globalMemoryEffects;

  GlobalMemoryEffects *getMemoryEffects(LoadOp load) {
    auto function = load->getParentOfType<FuncOp>();
    if (!function)
      return nullptr;
    return &globalMemoryEffects.try_emplace(function.getOperation(), function)
                .first->second;
  }

  // Read invariance is distinct from address invariance and safe speculation.
  // Alias groups supply only disjoint footprints; the shared effect summary
  // derives absence of writes. The masks and entry guards below remain
  // required.
  bool hasInvariantContents(Value pointer, const GlobalMemoryEffects &effects) {
    return effects.isExplicitlyImmutable(pointer) ||
           (effects.getAliasGroup(pointer) && !effects.mayModify(pointer));
  }

  // Entering a loop does not imply reaching every operation in its body. A
  // preceding region may diverge, so keep the new memory-contract path bounded
  // to loads with a straight-line prefix. This is deliberately conservative.
  bool hasStraightLinePrefix(LoadOp load) {
    for (Operation &op : *load->getBlock()) {
      if (&op == load.getOperation())
        return true;
      if (op.getNumRegions() != 0)
        return false;
    }
    return false;
  }

  // A bounded while-loop extension for constant initial conditions, such as
  // `offset = 0; while offset < HIDDEN_SIZE`. Do not speculate an ordinary load
  // out of a while loop whose first execution cannot be established.
  std::optional<bool> initialWhileCondition(scf::WhileOp loop) {
    // A constant condition does not establish that the before region reaches
    // its terminator: it could contain a nonterminating nested loop. Restrict
    // entry-test computation to operations safe to speculate themselves.
    for (Operation &op : loop.getBefore().front())
      if (!isa<scf::ConditionOp>(op) &&
          (op.getNumRegions() != 0 || !isSpeculatable(&op) ||
           !isMemoryEffectFree(&op)))
        return std::nullopt;
    auto condition = loop.getConditionOp().getCondition();
    auto resolveInitial = [&](Value value) {
      if (auto arg = dyn_cast<BlockArgument>(value))
        if (arg.getOwner() == &loop.getBefore().front())
          return loop.getInits()[arg.getArgNumber()];
      return value;
    };
    APInt constant;
    if (matchPattern(resolveInitial(condition), m_ConstantInt(&constant)))
      return !constant.isZero();
    auto cmp = condition.getDefiningOp<arith::CmpIOp>();
    if (!cmp || !cmp.getType().isInteger(1))
      return std::nullopt;
    APInt lhs, rhs;
    if (!matchPattern(resolveInitial(cmp.getLhs()), m_ConstantInt(&lhs)) ||
        !matchPattern(resolveInitial(cmp.getRhs()), m_ConstantInt(&rhs)))
      return std::nullopt;
    return arith::applyCmpPredicate(cmp.getPredicate(), lhs, rhs);
  }

  bool isLICMDisabled(LoopLikeOpInterface loopLike) {
    auto loopMD = loopLike->getAttrOfType<LLVM::LoopAnnotationAttr>(
        LLVM::LoopAnnotationAttr::name);
    if (!loopMD)
      return false;

    auto licmMD = loopMD.getLicm();
    if (!licmMD)
      return false;

    auto disable = licmMD.getDisable();
    return disable && disable.getValue();
  }

  bool isMemoryEffectFreeOrOnlyRead(Operation *op) {
    std::optional<SmallVector<MemoryEffects::EffectInstance>> effects =
        getEffectsRecursively(op);
    if (!effects)
      return false;
    return llvm::all_of(*effects,
                        [&](const MemoryEffects::EffectInstance &effect) {
                          return isa<MemoryEffects::Read>(effect.getEffect());
                        });
  }

  void runOnOperation() override {
    globalMemoryEffects.clear();
    isLoopMemoryEffectFreeOrOnlyRead.clear();
    // Walk through all loops in a function in innermost-loop-first order.
    // This way, we first LICM from the inner loop, and place the ops in the
    // outer loop, which in turn can be further LICM'ed.
    getOperation()->walk([&](LoopLikeOpInterface loopLike) {
      if (isLICMDisabled(loopLike))
        return;
      moveLoopInvariantCode(
          loopLike.getLoopRegions(),
          // isDefinedOutsideOfRegion
          [&](Value value, Region *region) {
            return loopLike.isDefinedOutsideOfLoop(value);
          },
          // shouldMoveOutOfRegion
          [&](Operation *op, Region *region) {
            auto load = dyn_cast<LoadOp>(op);
            if (!load)
              return isSpeculatable(op) && isMemoryEffectFree(op);
            if (load.getIsVolatile())
              return false;
            auto *effects = getMemoryEffects(load);
            bool readInvariant =
                enableMemoryContracts && effects &&
                hasInvariantContents(load.getPtr(), *effects) &&
                GlobalMemoryEffects::hasOnlyOrdinaryEffects(
                    loopLike,
                    effects->isExplicitlyImmutableGlobal(load.getPtr()));
            if (readInvariant && (load->getParentRegion() != region ||
                                  !hasStraightLinePrefix(load)))
              return false;
            if (auto whileOp =
                    dyn_cast<scf::WhileOp>(loopLike.getOperation())) {
              // Only after-region loads are conditional on the entry test.
              // Existing unannotated while behavior remains unchanged.
              if (!readInvariant || region != &whileOp.getAfter() ||
                  !initialWhileCondition(whileOp).has_value())
                return false;
            } else if (!isa<scf::ForOp>(loopLike.getOperation())) {
              return false;
            }
            if (!isLoopMemoryEffectFreeOrOnlyRead.contains(loopLike))
              isLoopMemoryEffectFreeOrOnlyRead[loopLike] =
                  isMemoryEffectFreeOrOnlyRead(loopLike);
            return isMemoryEffectFreeOrOnlyRead(op) &&
                   (readInvariant ||
                    isLoopMemoryEffectFreeOrOnlyRead[loopLike]);
          },
          // moveOutOfRegion
          [&](Operation *op, Region *) {
            // Create the new mask for load op.
            if (auto loadOp = dyn_cast<LoadOp>(op)) {
              IRRewriter rewriter(loopLike);
              Location loc = loopLike->getLoc();
              Value cond;
              if (auto forOp = dyn_cast<scf::ForOp>(loopLike.getOperation())) {
                auto predicate = forOp.getUnsignedCmp()
                                     ? arith::CmpIPredicate::ult
                                     : arith::CmpIPredicate::slt;
                cond = arith::CmpIOp::create(rewriter, loc, predicate,
                                             forOp.getLowerBound(),
                                             forOp.getUpperBound());
              } else if (auto whileOp =
                             dyn_cast<scf::WhileOp>(loopLike.getOperation())) {
                auto entry = initialWhileCondition(whileOp);
                if (!entry)
                  return;
                cond = arith::ConstantIntOp::create(rewriter, loc, *entry, 1);
              } else {
                return;
              }
              Value newMask = getPredMask(rewriter, loadOp.getPtr().getType(),
                                          loadOp.getMask(), cond);
              loadOp.getMaskMutable().assign(newMask);
            }
            loopLike.moveOutOfLoop(op);
          });
    });
  }
};

} // namespace mlir::triton
