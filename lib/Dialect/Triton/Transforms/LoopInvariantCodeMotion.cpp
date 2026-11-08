#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

namespace mlir::triton {

#define GEN_PASS_DEF_TRITONLOOPINVARIANTCODEMOTION
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

#define DEBUG_TYPE "triton-licm"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

// Keep pointer provenance deliberately small. Unknown casts, selections and
// loop-carried pointers must not inherit a caller's region contract.
BlockArgument getEntryArgument(Value ptr) {
  while (Operation *def = ptr.getDefiningOp()) {
    if (!isa<AddPtrOp, SplatOp, BroadcastOp, ExpandDimsOp, ReshapeOp>(def))
      return {};
    ptr = def->getOperand(0);
  }
  auto arg = dyn_cast<BlockArgument>(ptr);
  if (!arg)
    return {};
  auto func = dyn_cast<FuncOp>(arg.getOwner()->getParentOp());
  if (!func || arg.getOwner() != &func.getBody().front())
    return {};
  return arg;
}

bool hasArgumentFlag(BlockArgument arg, StringRef name) {
  auto func = cast<FuncOp>(arg.getOwner()->getParentOp());
  auto attr = func.getArgAttrOfType<IntegerAttr>(arg.getArgNumber(), name);
  return attr && attr.getType().isInteger(32) && attr.getInt() == 1;
}

std::optional<int64_t> getAliasGroup(BlockArgument arg) {
  auto func = cast<FuncOp>(arg.getOwner()->getParentOp());
  auto attr = func.getArgAttrOfType<IntegerAttr>(arg.getArgNumber(),
                                                 "tt.experimental_alias_group");
  if (!attr || !attr.getType().isInteger(32) || attr.getInt() < 0)
    return std::nullopt;
  return attr.getInt();
}

// A conservative argument-region effect summary. Group identities are local to
// this function: different nonnegative groups promise disjoint actual access
// regions; matching or missing groups may alias. The caller must establish the
// entire derived-access footprint, not merely compare base pointer addresses.
// This summary does not assert immutability. Clients ask whether the function
// may modify a region, then combine that answer with their own legality checks.
class ArgumentMemoryEffects {
public:
  explicit ArgumentMemoryEffects(FuncOp func) : func(func) {
    hasUnknownEffects =
        func->walk([&](Operation *op) {
              if (op == func.getOperation())
                return WalkResult::advance();
              if (auto load = dyn_cast<LoadOp>(op)) {
                if (load.getIsVolatile() || !getEntryArgument(load.getPtr()))
                  return WalkResult::interrupt();
                return WalkResult::advance();
              }
              if (auto store = dyn_cast<StoreOp>(op)) {
                auto arg = getEntryArgument(store.getPtr());
                if (!arg)
                  return WalkResult::interrupt();
                writtenArguments.push_back(arg);
                return WalkResult::advance();
              }
              if (isMemoryEffectFree(op) ||
                  op->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
                return WalkResult::advance();
              // Calls, atomics, async memory operations, assertions and unknown
              // effects require richer modeling before this analysis can reason
              // about them.
              return WalkResult::interrupt();
            })
            .wasInterrupted();
  }

  bool mayModify(Value ptr) {
    auto arg = getEntryArgument(ptr);
    if (hasUnknownEffects || !arg || arg.getOwner() != &func.getBody().front())
      return true;
    return llvm::any_of(writtenArguments, [&](BlockArgument written) {
      auto readGroup = getAliasGroup(arg);
      auto writeGroup = getAliasGroup(written);
      return !readGroup || !writeGroup || *readGroup == *writeGroup;
    });
  }

private:
  FuncOp func;
  SmallVector<BlockArgument> writtenArguments;
  bool hasUnknownEffects = false;
};

} // namespace

class LoopInvariantCodeMotionPass
    : public impl::TritonLoopInvariantCodeMotionBase<
          LoopInvariantCodeMotionPass> {

  DenseMap<LoopLikeOpInterface, bool> isLoopMemoryEffectFreeOrOnlyRead;
  DenseMap<Operation *, ArgumentMemoryEffects> argumentMemoryEffects;

  bool hasOnlyOrdinaryMemoryEffects(LoopLikeOpInterface loop) {
    return !loop->walk([](Operation *op) {
                  if (auto load = dyn_cast<LoadOp>(op))
                    return load.getIsVolatile() ? WalkResult::interrupt()
                                                : WalkResult::advance();
                  if (isa<StoreOp>(op) || isMemoryEffectFree(op) ||
                      op->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
                    return WalkResult::advance();
                  // In particular, do not move reads ahead of assertions or
                  // unknown calls.
                  return WalkResult::interrupt();
                })
                .wasInterrupted();
  }

  // Accept either an explicit invariant-content contract or derive a no-clobber
  // result from region relationships and visible effects. Neither proof alone
  // permits accessing an address outside the original masks and entry guards.
  bool hasInvariantContents(Value ptr) {
    auto arg = getEntryArgument(ptr);
    if (!arg)
      return false;
    // This explicit caller promise is stronger than readonly: memory reached
    // through the argument is initialized at entry and unchanged through all
    // aliases for the call. It is not inferred from race freedom.
    if (hasArgumentFlag(arg, "tt.experimental_immutable"))
      return true;
    // The weaker region contract supplies disjointness, not immutability or
    // initialization. Derive the absence of clobbers from visible effects.
    // Guards below preserve the original read's execution conditions. Requiring
    // an explicit group here leaves unannotated while-loop behavior unchanged.
    if (!getAliasGroup(arg))
      return false;
    auto func = cast<FuncOp>(arg.getOwner()->getParentOp());
    auto summary =
        argumentMemoryEffects.try_emplace(func.getOperation(), func).first;
    return !summary->second.mayModify(ptr);
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
    argumentMemoryEffects.clear();
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
            bool readInvariant = hasInvariantContents(load.getPtr()) &&
                                 hasOnlyOrdinaryMemoryEffects(loopLike);
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
