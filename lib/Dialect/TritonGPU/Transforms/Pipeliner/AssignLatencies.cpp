#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/GlobalMemoryEffects.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/MMAv5PipelineUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "triton-loop-pipeline"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir::triton::gpu {
namespace {

//===----------------------------------------------------------------------===//
// assignLatencies
//===----------------------------------------------------------------------===//

bool hasLoopCarriedAccumulatorCycle(ttng::MMAv5OpInterface mma,
                                    scf::ForOp forOp) {
  Value start = mma.getAccDep();
  Value target = mma.getToken();
  if (!start || !target)
    return false;

  // Follow the accumulator dependency backwards, mapping loop block arguments
  // to their yielded values. A cross-MMA cycle is found when the recurrence
  // returns to this MMA through another MMA.
  SmallVector<std::pair<Value, bool>> worklist = {{start, false}};
  // Following all operands may reach the same value along paths that differ in
  // whether another MMA was crossed. Track those states separately so that a
  // visit before crossing another MMA does not suppress a visit after crossing
  // one.
  DenseSet<Value> seenBeforeAnotherMMA;
  DenseSet<Value> seenAfterAnotherMMA;
  while (!worklist.empty()) {
    auto [current, crossedAnotherMMA] = worklist.pop_back_val();
    DenseSet<Value> &seen =
        crossedAnotherMMA ? seenAfterAnotherMMA : seenBeforeAnotherMMA;
    if (!seen.insert(current).second)
      continue;
    if (current == target) {
      if (crossedAnotherMMA)
        return true;
      continue;
    }

    if (auto arg = dyn_cast<BlockArgument>(current)) {
      if (arg.getOwner() == forOp.getBody() && arg.getArgNumber() > 0)
        worklist.emplace_back(forOp.getYieldedValues()[arg.getArgNumber() - 1],
                              crossedAnotherMMA);
      continue;
    }

    Operation *def = current.getDefiningOp();
    if (!def || !forOp->isAncestor(def))
      continue;
    crossedAnotherMMA |=
        isa<ttng::MMAv5OpInterface>(def) && def != mma.getOperation();
    for (Value operand : getNestedOperands(def))
      worklist.emplace_back(operand, crossedAnotherMMA);
  }
  return false;
}

// Return true if the preconditions for pipelining the loop are met.
bool preCondition(scf::ForOp forOp) {
  // Skip loop with distance > 1 for now.
  // TODO: relax the constraint in the expander.
  if (loopHasDistGreaterThanOne(forOp))
    return false;
  // Don't pipeline outer loops.
  if (isOuterLoop(forOp))
    return false;
  return true;
}

bool hasLatenciesAssigned(scf::ForOp forOp) {
  auto helper = TritonDialect::getLoaded(forOp)->getLatencyAttrHelper();
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (helper.getAttr(&op))
      return true;
  }
  return false;
}

void assignUserProvidedLatencies(scf::ForOp forOp,
                                 DenseMap<Operation *, int> &opLatency) {
  auto helper = TritonDialect::getLoaded(forOp)->getLatencyAttrHelper();
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (auto latencyAttr = helper.getAttr(&op)) {
      opLatency[&op] = latencyAttr.getInt();
    }
  }
}

class AssignLoadLatencies {
public:
  AssignLoadLatencies(scf::ForOp forOp, int numStages,
                      DenseMap<Operation *, int> &opLatency)
      : forOp(forOp), numStages(numStages), opLatency(opLatency) {};

  void run() {
    bool pipelineWithoutDot = forOp->hasAttr(mlir::triton::kNumStagesAttrName);
    ModuleOp moduleOp = forOp->getParentOfType<ModuleOp>();
    tt::ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);

    llvm::MapVector<Operation *, std::pair<int, Operation *>> loadOpToIndLevel =
        loadOpsToIndirectionLevel(forOp, pipelineWithoutDot, axisInfoAnalysis,
                                  numStages);
    if (loadOpToIndLevel.empty())
      return;

    // Calculate the stage distance between applicable loads.
    int maxIndirectionLevel = 0;
    for (auto &[loadOp, info] : loadOpToIndLevel)
      maxIndirectionLevel = std::max(maxIndirectionLevel, info.first);
    unsigned loadLatency = (numStages - 1) / (maxIndirectionLevel + 1);

    for (auto [loadOp, dist] : loadOpToIndLevel) {
      opLatency[loadOp] = loadLatency;
    }
  }

private:
  scf::ForOp forOp;
  int numStages;
  DenseMap<Operation *, int> &opLatency;

public:
  static bool canHaveSharedEncoding(tt::LoadOp op) {
    // If used by an user with DotOp encoding, all the uses must be compatible.
    bool incompatible = false;
    getSharedEncIfAllUsersAreDotEnc(op.getResult(), incompatible);
    return !incompatible;
  }

  static bool
  isPipeliningBeneficial(Operation *op, Operation *finalUser,
                         tt::ModuleAxisInfoAnalysis &axisInfoAnalysis,
                         bool filterSmall) {
    if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
      if (filterSmall && !canBeConvertedToAsyncLoad(loadOp, axisInfoAnalysis)) {
        LDBG("Load " << *loadOp << " is too small for pipelining");
        return false;
      }
    }
    if (auto loadOp = dyn_cast<tt::DescriptorLoadLikeOpInterface>(op)) {
      auto descTy = cast<tt::TensorDescType>(loadOp.getDesc().getType());
      if (descTy.getSharedLayout() && !canPipelineTMALoad(op)) {
        LDBG("TMA load " << *op
                         << " has a per-stage allocation that is not aligned "
                            "for pipelining");
        return false;
      }
      return true;
    }
    if (!canHaveSharedEncoding(cast<tt::LoadOp>(op))) {
      LDBG("Load " << *op << " cannot have shared encoding");
      return false;
    }

    ttg::SharedEncodingTrait localAllocEnc;
    if (llvm::any_of(op->getUsers(), [&](Operation *user) {
          return isa<ttg::LocalAllocOp>(user);
        })) {
      for (auto user : op->getUsers()) {
        auto localAlloc = dyn_cast<ttg::LocalAllocOp>(user);
        if (!localAlloc)
          continue;
        auto enc = mlir::cast<ttg::SharedEncodingTrait>(
            localAlloc.getType().getEncoding());
        if (!localAllocEnc) {
          localAllocEnc = enc;
        }
        if (enc != localAllocEnc) {
          // If the load is used by a LocalAllocOp, all the users need to have
          // the same encoding.
          return false;
        }
      }
    }

    if (localAllocEnc) {
      auto registerTy = cast<RankedTensorType>(op->getResultTypes()[0]);
      auto vecBytes = getCopyVecBytes(registerTy, localAllocEnc);
      if (filterSmall && vecBytes < 4) {
        // At least 4 bytes need to be consecutive for cp.async
        return false;
      }
    }

    return true;
  }
};

class AssignMMALatencies {
public:
  AssignMMALatencies(scf::ForOp forOp, DenseMap<Operation *, int> &opLatency)
      : forOp(forOp), opLatency(opLatency) {};

  void run() {
    DenseMap<Operation *, int> mmaSelfLatency;
    // Check if the load op (mma operand) is pipelineable.
    auto isLoadToBePipelined = [&](Operation *op) {
      return opLatency.contains(op) && opLatency[op] > 0;
    };
    for (auto &op : forOp.getBody()->without_terminator()) {
      // If the acc can not be multibuffered, do not pipeline the uses of
      // the MMA to later stages.
      if (auto mma = dyn_cast<ttng::MMAv5OpInterface>(&op)) {
        // Try to push out the wait by one stage even if the operands are not
        // pipelineable, but we know where the loads are scheduled, so we can
        // place the wait right before the loads.

        if (hasSyncDots(forOp)) {
          // Skip pipelining MMA in the loops where sync dots are used. This
          // is a dirty heuristic for performance drops in kernels where we
          // would rather want to have last iteration peeled instead of having a
          // full iteration of masked operations only to execute single wait.
          continue;
        }
        auto pipeHelper = ttng::MMAv5PipelineableOperandsHelper(
            mma, forOp, isLoadToBePipelined);
        if (pipeHelper.isPipelineable ||
            (pipeHelper.isOperandsStateDetermined &&
             !ttng::hasLoadsAfterMMA(mma, forOp))) {
          // MMA can be overlapped with itself
          mmaSelfLatency[mma] = 1;
          if (!ttng::requiresAccMultiBuffering(mma, forOp) ||
              (ttng::isAccMultibufferingPossible(mma, forOp) &&
               !getDisallowAccMultiBuffer(forOp))) {
            // MMA's users can be pushed to the next stage
            opLatency[&op] = 1;
          }
          // HACK: A pipelined MMA's latency should equal the number of buffers
          // for the accumulator, but when the user is in an `scf.if` in SWP,
          // the `scf.if` is pushed to the end of the loop rather than peeled
          // before the MMA op, requiring an extra buffer due to liverange
          // overlap. WS does not have this problem because the MMA is placed in
          // a different partition than the MMA, so we can correctly set the
          // latency.
          if (isWarpSpecialized(forOp)) {
            if (ttng::hasAccReadModifyWrite(mma, forOp))
              opLatency.erase(&op); // can't pipeline the MMA
            else
              opLatency[&op] += 1;
            // If all inputs to the MMA are warp specialized, set the self
            // latency to 0 since the MMA won't need to wait on itself.
            auto cantWarpSpec = [](Operation *op) { return isa<LoadOp>(op); };
            auto warpSpecHelper = ttng::MMAv5PipelineableOperandsHelper(
                mma, forOp, [&](Operation *op) {
                  return isLoadToBePipelined(op) && !cantWarpSpec(op);
                });
            if (warpSpecHelper.isPipelineable ||
                (warpSpecHelper.isOperandsStateDetermined &&
                 llvm::none_of(warpSpecHelper.unpipelineableOperandDefs,
                               cantWarpSpec)))
              mmaSelfLatency[mma] = 0;
          }
          if (hasLoopCarriedAccumulatorCycle(mma, forOp))
            opLatency.erase(&op);
        }
      }
    }
    serializeSelfLatencies(forOp->getParentOfType<ModuleOp>(), mmaSelfLatency);
  }

private:
  scf::ForOp forOp;
  DenseMap<Operation *, int> &opLatency;

  bool hasSyncDots(scf::ForOp forOp) {
    for (auto &op : forOp.getBody()->without_terminator()) {
      if (isa<mlir::triton::DotOp>(op))
        return true;
    }
    return false;
  }

  bool isWarpSpecialized(scf::ForOp forOp) {
    // Automatic warp specialization is disabled for two-CTA kernels.
    if (ttng::getModuleTwoCTAs(forOp))
      return false;
    scf::ForOp current = forOp;
    do {
      if (current->hasAttr(kWarpSpecializeAttrName)) {
        return true;
      }
      current = current->getParentOfType<scf::ForOp>();
    } while (current);
    return false;
  };
};

// Discover operations that should become async and assign latencies to them
// based on the numStages value provided by the user.
//
// Look for load ops that directly or indirectly feed into dot ops. Based on the
// requested number of stages assign the latencies in a way that cover all the
// stages with the sum of latencies in the chain from the first load to the
// final dot op.
void assignLatencies(ModuleOp moduleOp, int defaultNumStages) {
  SmallVector<scf::ForOp> loops;
  moduleOp->walk([&](scf::ForOp forOp) {
    // Bail out for loops with num_stage <= 1.
    if (preCondition(forOp) &&
        getNumStagesOrDefault(forOp, defaultNumStages) > 1)
      loops.push_back(forOp);
  });
  if (loops.empty())
    return;

  DenseMap<Operation *, int> opLatency;
  for (auto forOp : loops) {
    if (hasLatenciesAssigned(forOp)) {
      assignUserProvidedLatencies(forOp, opLatency);
      continue;
    }
    int numStages = getNumStagesOrDefault(forOp, defaultNumStages);
    AssignLoadLatencies(forOp, numStages, opLatency).run();
    AssignMMALatencies(forOp, opLatency).run();
  }
  serializeLatencies(moduleOp, opLatency);
}

// This opt-in policy adds a scheduling candidate only when the caller supplied
// memory facts for every ordinary read. Profitability is chosen by the caller's
// variant selection; the annotation itself is not a performance guarantee.
void inferContractPipelineStages(ModuleOp module, int stages) {
  if (stages <= 1)
    return;
  module.walk([&](scf::ForOp loop) {
    if (!preCondition(loop) || loop->hasAttr(kNumStagesAttrName) ||
        loop->hasAttr(kWarpSpecializeAttrName) ||
        loop->hasAttr(kScheduledMaxStageAttrName) || hasLatenciesAssigned(loop))
      return;
    APInt step;
    if (!matchPattern(loop.getStep(), m_ConstantInt(&step)) || !step.isOne())
      return;
    APInt lower, upper;
    if (matchPattern(loop.getLowerBound(), m_ConstantInt(&lower)) &&
        matchPattern(loop.getUpperBound(), m_ConstantInt(&upper))) {
      // A zero- or one-trip loop has no useful next iteration to prefetch.
      // Widen before adding one so this test also works at the integer limits.
      // Respect scf.for's signed/unsigned comparison rather than interpreting
      // the bounds as host integers. Explicit source schedules were excluded
      // above and remain unchanged by this automatic profitability gate.
      unsigned width = lower.getBitWidth() + 1;
      lower = loop.getUnsignedCmp() ? lower.zext(width) : lower.sext(width);
      upper = loop.getUnsignedCmp() ? upper.zext(width) : upper.sext(width);
      APInt afterOne = lower + 1;
      if (loop.getUnsignedCmp() ? upper.ule(afterOne) : upper.sle(afterOne))
        return;
    }
    auto function = loop->getParentOfType<tt::FuncOp>();
    if (!function)
      return;
    tt::GlobalMemoryEffects memory(function);
    bool hasLoad = false;
    for (Operation &op : loop.getBody()->without_terminator()) {
      // Keep source branches, nested regions, explicit schedules and existing
      // async protocols outside this first automatic policy.
      if ((op.getNumRegions() && !isa<tt::ReduceOp>(op)) ||
          op.hasAttr(kLoopStageAttrName) ||
          op.hasAttr(kLoopClusterAttrName))
        return;
      if (auto load = dyn_cast<tt::LoadOp>(op)) {
        hasLoad = true;
        auto pointer = dyn_cast<tt::PointerType>(
            getElementTypeOrSelf(load.getPtr().getType()));
        if (!pointer || pointer.getAddressSpace() != tt::PtrAddrSpace::Global ||
            load.getIsVolatile() ||
            (!memory.isExplicitlyImmutableGlobal(load.getPtr()) &&
             !memory.getAliasGroup(load.getPtr())))
          return;
      } else if (!isa<tt::StoreOp>(op) && !isMemoryEffectFree(&op)) {
        return;
      }
    }
    if (!hasLoad || !memory.hasIndependentGlobalMemoryAccesses(loop))
      return;
    auto attr = IntegerAttr::get(IntegerType::get(module.getContext(), 32), stages);
    loop->setAttr(kNumStagesAttrName, attr);
    loop->setAttr("tt.experimental_contract_pipeline", attr);
  });
}

} // namespace

// Create a map from load ops to their indirection level and the
// final use of the load op (another load op, or a dot op).
// Indirection level is "0" for the load op directly used by the dot op,
// "1" for the load op used by the load op used by the dot op, and so on.
llvm::MapVector<Operation *, std::pair<int, Operation *>>
loadOpsToIndirectionLevel(scf::ForOp forOp, bool pipelineWithoutDot,
                          tt::ModuleAxisInfoAnalysis &axisInfoAnalysis,
                          int numStages, bool filterSmall) {
  llvm::MapVector<Operation *, std::pair<int, Operation *>> loadOpToIndLevel;
  DenseSet<Operation *> seen;
  DenseSet<Operation *> excluded;

  std::function<void(Operation *, Operation *, int)> dfs =
      [&](Operation *op, Operation *finalUser, int distance) {
        if (!seen.insert(op).second || excluded.contains(op))
          return;
        if (isa<tt::LoadOp, tt::DescriptorLoadLikeOpInterface>(op)) {
          if (!AssignLoadLatencies::isPipeliningBeneficial(
                  op, finalUser, axisInfoAnalysis, filterSmall))
            return;
          if (loadOpToIndLevel.contains(op)) {
            int level = loadOpToIndLevel[op].first;
            if (level != distance) {
              // If we have multiple uses at different distances, we don't
              // know which one to pick.
              LDBG("Load " << *op
                           << " has multiple uses at different distances:"
                           << level << " and " << distance);
              loadOpToIndLevel.erase(op);
              excluded.insert(op);
              return;
            }
          } else {
            LDBG("Load " << *op << " considered for pipelining with distance "
                         << distance);
            loadOpToIndLevel[op] = {distance, finalUser};
          }
          finalUser = op;
          distance++;
        }
        for (Value operand : getNestedOperands(op)) {
          if (isa<mlir::triton::DotOpInterface>(op)) {
            // Heuristic: only pipeline A and B operands of the dot op.
            if (operand == op->getOperand(2))
              continue;
          }
          Value v = operand;
          Operation *defOp = v.getDefiningOp();
          if (defOp && defOp->getBlock() == op->getBlock()) {
            dfs(defOp, finalUser, distance);
          }
        }
      };

  for (Operation &op : forOp.getBody()->without_terminator()) {
    // Arbitrary heuristic. TMEMStoreOp is included to keep logic consistent
    // with legacy code when we weren't hoisting tmem allocas.
    if (!isa<mlir::triton::DotOpInterface, ttng::TMEMStoreOp>(op))
      continue;
    seen.clear();
    dfs(&op, &op, 0);
  }

  // If the loop has numStages attribute, also consider pipelining other loads
  // that are not directly used by dot ops.
  if (pipelineWithoutDot) {
    for (Operation &op : forOp.getBody()->without_terminator()) {
      if (!isa<tt::LoadOp, tt::DescriptorLoadLikeOpInterface>(op))
        dfs(&op, &op, 0);
    }
  }

  // We assume loads with different dist are assigned to different stages.
  // If numStages is 2, we will have no stage available for indirect loads
  // with dist >= 1. In general, when dist is equal to numStages - 1, we
  // should not pipeline it.
  for (auto iter = loadOpToIndLevel.begin(); iter != loadOpToIndLevel.end();) {
    if (iter->second.first >= numStages - 1)
      iter = loadOpToIndLevel.erase(iter);
    else
      ++iter;
  }

  return loadOpToIndLevel;
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

#define GEN_PASS_DEF_TRITONGPUASSIGNLATENCIES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

struct AssignLatencies
    : public impl::TritonGPUAssignLatenciesBase<AssignLatencies> {
  using TritonGPUAssignLatenciesBase::TritonGPUAssignLatenciesBase;

  void runOnOperation() override {
    // Establish a proof before compiler-owned protocols are introduced.
    // Latencies and source hints remain provisional: PartitionScheduling and
    // the software pipeliner check legality before transforming the loop.
    getOperation().walk([&](LoopLikeOpInterface loop) {
      loop->removeAttr(kGlobalPipelineProofAttrName);
      if (hasSafeGlobalPipelineAccesses(loop.getOperation()))
        loop->setAttr(kGlobalPipelineProofAttrName, UnitAttr::get(&getContext()));
    });
    inferContractPipelineStages(getOperation(), contractPipelineStages);
    if (forceTmemSingleBuffer) {
      getOperation().walk([&](scf::ForOp loop) {
        if (!loop->hasAttr(kWarpSpecializeAttrName))
          return;
        // Explicit latencies bypass inference. Do not reduce buffering while
        // keeping a schedule inferred for a different buffer count. Include
        // nested loops because Aref allocation reads the enclosing WS policy.
        WalkResult explicitLatencies = loop.walk([&](scf::ForOp nestedLoop) {
          return hasLatenciesAssigned(nestedLoop) ? WalkResult::interrupt()
                                                 : WalkResult::advance();
        });
        if (explicitLatencies.wasInterrupted())
          return;
        loop->setAttr(kDisallowAccMultiBufferAttrName,
                      UnitAttr::get(&getContext()));
      });
    }
    // Apply the source-level policy before latency analysis so scheduling and
    // Aref buffer allocation agree on the allowed accumulator overlap.
    assignLatencies(getOperation(), numStages);
  }
};

} // namespace mlir::triton::gpu
