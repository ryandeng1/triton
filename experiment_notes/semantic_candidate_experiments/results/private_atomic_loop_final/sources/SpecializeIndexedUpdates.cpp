#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

#include <array>
#include <limits>

namespace mlir::triton {

#define GEN_PASS_DEF_TRITONSPECIALIZEINDEXEDUPDATES
#define GEN_PASS_DEF_TRITONRELAXPRIVATEATOMICS
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

std::optional<int64_t> argumentInteger(BlockArgument arg, StringRef name) {
  auto function = dyn_cast<FuncOp>(arg.getOwner()->getParentOp());
  if (!function || arg.getOwner() != &function.getBody().front())
    return std::nullopt;
  auto attr = function.getArgAttrOfType<IntegerAttr>(arg.getArgNumber(), name);
  if (!attr || !attr.getType().isInteger(32))
    return std::nullopt;
  return attr.getInt();
}

bool argumentFlag(BlockArgument arg, StringRef name) {
  return argumentInteger(arg, name) == 1;
}

bool isRepresentationChange(Operation *op) {
  return isa<SplatOp, BroadcastOp, ExpandDimsOp, ReshapeOp>(op) ||
         op->getName().getStringRef() == "ttg.convert_layout";
}

BlockArgument pointerRoot(Value pointer) {
  while (Operation *op = pointer.getDefiningOp()) {
    if (!isa<AddPtrOp>(op) && !isRepresentationChange(op))
      return {};
    pointer = op->getOperand(0);
  }
  auto argument = dyn_cast<BlockArgument>(pointer);
  if (!argument || !isa<PointerType>(argument.getType()))
    return {};
  auto function = dyn_cast<FuncOp>(argument.getOwner()->getParentOp());
  if (!function || argument.getOwner() != &function.getBody().front())
    return {};
  return argument;
}

std::optional<int64_t> integerConstant(Value value) {
  Attribute attr;
  if (!matchPattern(value, m_Constant(&attr)))
    return std::nullopt;
  APInt integer;
  if (auto scalar = dyn_cast<IntegerAttr>(attr)) {
    integer = scalar.getValue();
  } else if (auto tensor = dyn_cast<DenseIntElementsAttr>(attr)) {
    if (!tensor.isSplat())
      return std::nullopt;
    integer = tensor.getSplatValue<APInt>();
  } else {
    return std::nullopt;
  }
  if (!integer.isSignedIntN(64))
    return std::nullopt;
  return integer.getSExtValue();
}

// A deliberately bounded nonnegative affine expression. INDEX denotes the
// immutable unique value loaded at IndexArg[program_id(x)], not an arbitrary
// loaded value. LANE denotes one logical make_range(0, blockSize) element.
struct Offset {
  enum Term { INDEX, X, Y, LANE, NUM_TERMS };
  std::array<int64_t, NUM_TERMS> coefficients{};
  int64_t constant = 0;

  bool isConstant() const {
    return llvm::all_of(coefficients, [](int64_t c) { return c == 0; });
  }

  bool operator==(const Offset &) const = default;
};

std::optional<Offset> add(Offset left, const Offset &right) {
  if (llvm::AddOverflow(left.constant, right.constant, left.constant))
    return std::nullopt;
  for (unsigned i = 0; i < Offset::NUM_TERMS; ++i)
    if (llvm::AddOverflow(left.coefficients[i], right.coefficients[i],
                          left.coefficients[i]))
      return std::nullopt;
  return left;
}

std::optional<Offset> multiply(Offset value, int64_t factor) {
  if (factor < 0 || llvm::MulOverflow(value.constant, factor, value.constant))
    return std::nullopt;
  for (int64_t &coefficient : value.coefficients)
    if (llvm::MulOverflow(coefficient, factor, coefficient))
      return std::nullopt;
  return value;
}

class IndexedAddressProof {
public:
  IndexedAddressProof(int64_t gridX, int64_t gridY, int64_t blockSize)
      : gridX(gridX), gridY(gridY), blockSize(blockSize) {}

  bool prove(Value pointer, Value mask, BlockArgument destination) {
    auto offset = pointerOffset(pointer, destination);
    if (!offset || !indexArgument)
      return false;
    int64_t width = offset->coefficients[Offset::INDEX];
    if (width <= 0 || offset->coefficients[Offset::X] != 0 ||
        offset->coefficients[Offset::Y] != blockSize ||
        offset->coefficients[Offset::LANE] != 1 || offset->constant != 0)
      return false;
    // This exact launch contract also prevents forgotten grid dimensions from
    // repeating the same update. z=1 is checked before constructing the proof.
    if (gridY != llvm::divideCeil(width, blockSize))
      return false;
    auto maximumOffset = maximum(*offset);
    if (!maximumOffset ||
        *maximumOffset > std::numeric_limits<int64_t>::max() / 4)
      return false;
    Offset column;
    column.coefficients[Offset::Y] = blockSize;
    column.coefficients[Offset::LANE] = 1;
    return width % blockSize == 0 ||
           maskImpliesColumnBound(mask, column, width);
  }

private:
  std::optional<int64_t> maximum(const Offset &offset) const {
    std::array<int64_t, Offset::NUM_TERMS> maxima{
        indexBound > 0 ? indexBound - 1 : 0, gridX - 1, gridY - 1,
        blockSize - 1};
    int64_t result = offset.constant;
    for (unsigned i = 0; i < Offset::NUM_TERMS; ++i) {
      int64_t term;
      if (llvm::MulOverflow(offset.coefficients[i], maxima[i], term) ||
          llvm::AddOverflow(result, term, result))
        return std::nullopt;
    }
    return result;
  }

  bool fits(const Offset &offset, Type type) const {
    auto integer = dyn_cast<IntegerType>(getElementTypeOrSelf(type));
    // Keep the proof within the integer arithmetic used by the existing
    // kernels.
    if (!integer || (integer.getWidth() != 32 && integer.getWidth() != 64))
      return false;
    auto upper = maximum(offset);
    if (!upper)
      return false;
    int64_t limit = integer.getWidth() == 32
                        ? std::numeric_limits<int32_t>::max()
                        : std::numeric_limits<int64_t>::max();
    return offset.constant >= 0 && *upper <= limit;
  }

  std::optional<Offset> expression(Value value) {
    Operation *op = value.getDefiningOp();
    if (!op)
      return std::nullopt;
    std::optional<Offset> result;
    if (auto constant = integerConstant(value)) {
      if (*constant < 0)
        return std::nullopt;
      result = Offset{};
      result->constant = *constant;
    } else if (auto pid = dyn_cast<GetProgramIdOp>(op)) {
      result = Offset{};
      if (pid.getAxis() == ProgramIDDim::X)
        result->coefficients[Offset::X] = 1;
      else if (pid.getAxis() == ProgramIDDim::Y)
        result->coefficients[Offset::Y] = 1;
      // The launch's z extent is exactly one, so program_id(z) is zero.
    } else if (auto range = dyn_cast<MakeRangeOp>(op)) {
      auto type = dyn_cast<RankedTensorType>(range.getType());
      if (range.getStart() != 0 || range.getEnd() != blockSize || !type ||
          type.getRank() != 1 || type.getDimSize(0) != blockSize)
        return std::nullopt;
      result = Offset{};
      result->coefficients[Offset::LANE] = 1;
    } else if (auto load = dyn_cast<LoadOp>(op)) {
      result = indexLoad(load);
    } else if (isRepresentationChange(op) ||
               isa<arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp>(op)) {
      // Checking both the operand and result ranges makes these integer casts
      // lossless; in particular a truncating index cast needs a fitting bound.
      result = expression(op->getOperand(0));
    } else if (auto sum = dyn_cast<arith::AddIOp>(op)) {
      auto left = expression(sum.getLhs());
      auto right = expression(sum.getRhs());
      if (left && right)
        result = add(*left, *right);
    } else if (auto product = dyn_cast<arith::MulIOp>(op)) {
      auto left = expression(product.getLhs());
      auto right = expression(product.getRhs());
      if (left && right) {
        if (left->isConstant())
          result = multiply(*right, left->constant);
        else if (right->isConstant())
          result = multiply(*left, right->constant);
      }
    }
    if (!result || !fits(*result, value.getType()))
      return std::nullopt;
    return result;
  }

  std::optional<Offset> pointerOffset(Value pointer,
                                      BlockArgument expectedRoot) {
    if (pointer == expectedRoot)
      return Offset{};
    Operation *op = pointer.getDefiningOp();
    if (!op)
      return std::nullopt;
    if (isRepresentationChange(op))
      return pointerOffset(op->getOperand(0), expectedRoot);
    auto increment = dyn_cast<AddPtrOp>(op);
    if (!increment)
      return std::nullopt;
    auto base = pointerOffset(increment.getPtr(), expectedRoot);
    auto offset = expression(increment.getOffset());
    if (!base || !offset)
      return std::nullopt;
    return add(*base, *offset);
  }

  std::optional<Offset> indexLoad(LoadOp load) {
    if (load.getIsVolatile() || load.getMask() ||
        !isa<IntegerType>(load.getType()))
      return std::nullopt;
    auto root = pointerRoot(load.getPtr());
    if (!root || (indexArgument && indexArgument != root))
      return std::nullopt;
    auto count = argumentInteger(root, "tt.experimental_unique_indices");
    auto bound = argumentInteger(root, "tt.experimental_index_bound");
    if (!count || !bound || *count <= 0 || *bound <= 0 || *count > *bound ||
        gridX > *count)
      return std::nullopt;
    // Index addresses must be plain IndexArg[program_id(x)]. Reject indirection
    // rather than recursively accepting another loaded value as its offset.
    if (resolvingIndexAddress)
      return std::nullopt;
    resolvingIndexAddress = true;
    auto address = pointerOffset(load.getPtr(), root);
    resolvingIndexAddress = false;
    Offset expected;
    expected.coefficients[Offset::X] = 1;
    if (!address || *address != expected)
      return std::nullopt;
    indexArgument = root;
    indexBound = *bound;
    Offset result;
    result.coefficients[Offset::INDEX] = 1;
    return result;
  }

  bool maskImpliesColumnBound(Value mask, const Offset &column, int64_t width) {
    if (!mask)
      return false;
    Operation *op = mask.getDefiningOp();
    if (!op)
      return false;
    if (isRepresentationChange(op))
      return maskImpliesColumnBound(op->getOperand(0), column, width);
    if (auto conjunction = dyn_cast<arith::AndIOp>(op))
      return maskImpliesColumnBound(conjunction.getLhs(), column, width) ||
             maskImpliesColumnBound(conjunction.getRhs(), column, width);
    auto comparison = dyn_cast<arith::CmpIOp>(op);
    if (!comparison ||
        (comparison.getPredicate() != arith::CmpIPredicate::slt &&
         comparison.getPredicate() != arith::CmpIPredicate::ult))
      return false;
    auto left = expression(comparison.getLhs());
    auto right = expression(comparison.getRhs());
    return left && right && *left == column && right->isConstant() &&
           right->constant == width;
  }

  int64_t gridX;
  int64_t gridY;
  int64_t blockSize;
  int64_t indexBound = 0;
  BlockArgument indexArgument;
  bool resolvingIndexAddress = false;
};

bool closedPrivateEffects(FuncOp function, AtomicRMWOp candidate,
                          BlockArgument destination) {
  auto outputGroup =
      argumentInteger(destination, "tt.experimental_alias_group");
  if (!outputGroup || *outputGroup < 0)
    return false;
  for (Operation &op : function.getBody().front()) {
    if (&op == candidate.getOperation())
      continue;
    if (op.getNumRegions() != 0 ||
        isa<CallOp, ElementwiseInlineAsmOp, ExternElementwiseOp>(op))
      return false;
    if (auto load = dyn_cast<LoadOp>(op)) {
      auto root = pointerRoot(load.getPtr());
      if (load.getIsVolatile() || !root)
        return false;
      auto group = argumentInteger(root, "tt.experimental_alias_group");
      if (!group || *group < 0 || *group == *outputGroup)
        return false;
      continue;
    }
    if (!isMemoryEffectFree(&op))
      return false;
  }
  return true;
}

bool isCompleteKernelBody(FuncOp function) {
  if (!function.isPublic() || function.getBody().empty() ||
      !llvm::hasSingleElement(function.getBody()))
    return false;
  // This is a whole-kernel effects proof. Do not apply it to a helper whose
  // caller can have memory effects outside the inspected function body.
  auto uses = SymbolTable::getSymbolUses(function, function->getParentOp());
  return uses && uses->empty();
}

// Integer addition modulo 2^32 has an order-independent final value. This
// permits repeated updates across iterations and atomic sites, even when the
// original acquire/release operations constrain their cross-location order.
// Floating-point addition lacks this property and retains the separate,
// single-site straight-line proof below.
bool relaxPrivateIntegerAtomics(FuncOp function) {
  SmallVector<AtomicRMWOp> candidates;
  function.walk([&](AtomicRMWOp atomic) { candidates.push_back(atomic); });
  if (candidates.empty())
    return false;
  auto destination = pointerRoot(candidates.front().getPtr());
  if (!destination ||
      cast<PointerType>(destination.getType()).getAddressSpace() !=
          PtrAddrSpace::Global ||
      !argumentFlag(destination, "tt.experimental_private") ||
      !argumentFlag(destination, "tt.experimental_num_ctas"))
    return false;
  auto outputGroup =
      argumentInteger(destination, "tt.experimental_alias_group");
  if (!outputGroup || *outputGroup < 0)
    return false;

  bool changesOrdering = false;
  for (AtomicRMWOp atomic : candidates) {
    auto type = dyn_cast<RankedTensorType>(atomic.getType());
    if (!type || type.getRank() != 1 || !type.getElementType().isInteger(32) ||
        type.getDimSize(0) <= 0 || atomic.getAtomicRmwOp() != RMWOp::ADD ||
        atomic.getScope() != MemSyncScope::GPU ||
        !atomic.getResult().use_empty() ||
        pointerRoot(atomic.getPtr()) != destination)
      return false;
    switch (atomic.getSem()) {
    case MemSemantic::ACQUIRE:
    case MemSemantic::RELEASE:
    case MemSemantic::ACQUIRE_RELEASE:
      changesOrdering = true;
      break;
    case MemSemantic::RELAXED:
      break;
    default:
      return false;
    }
  }
  if (!changesOrdering)
    return false;

  auto effects = function.walk([&](Operation *op) -> WalkResult {
    if (op == function.getOperation() || isa<AtomicRMWOp>(op))
      return WalkResult::advance();
    // The induction variable may affect addresses, values, masks, and nested
    // loop bounds. No atomic result or output load can affect this control
    // flow. Restrict this initial extension to for loops without carried state;
    // in particular, do not infer pointer roots through loop-carried arguments.
    if (auto loop = dyn_cast<scf::ForOp>(op))
      return loop.getNumRegionIterArgs() == 0 ? WalkResult::advance()
                                              : WalkResult::interrupt();
    if (op->getNumRegions() != 0 ||
        isa<CallOp, ElementwiseInlineAsmOp, ExternElementwiseOp>(op))
      return WalkResult::interrupt();
    if (auto load = dyn_cast<LoadOp>(op)) {
      auto root = pointerRoot(load.getPtr());
      if (load.getIsVolatile() || !root)
        return WalkResult::interrupt();
      auto group = argumentInteger(root, "tt.experimental_alias_group");
      return group && *group >= 0 && *group != *outputGroup
                 ? WalkResult::advance()
                 : WalkResult::interrupt();
    }
    return isMemoryEffectFree(op) ? WalkResult::advance()
                                  : WalkResult::interrupt();
  });
  if (effects.wasInterrupted())
    return false;

  // Validate the entire invocation before changing any site. Output accesses
  // consist exclusively of these unused-result additions; disjoint ordinary
  // reads cannot observe them. The private contract excludes external partners.
  // Removing ordering preserves the updates executed and their modular sum,
  // including overflow. No operation is moved or removed.
  for (AtomicRMWOp atomic : candidates)
    atomic.setSem(MemSemantic::RELAXED);
  function->setAttr(
      "tt.experimental_private_atomic_relaxed",
      IntegerAttr::get(IntegerType::get(function.getContext(), 32), 1));
  return true;
}

bool relaxPrivateAtomic(FuncOp function) {
  if (!isCompleteKernelBody(function))
    return false;
  if (relaxPrivateIntegerAtomics(function))
    return true;
  AtomicRMWOp candidate;
  for (Operation &op : function.getBody().front()) {
    if (auto atomic = dyn_cast<AtomicRMWOp>(op)) {
      if (candidate)
        return false;
      candidate = atomic;
    }
  }
  if (!candidate || !candidate.getResult().use_empty() ||
      candidate.getAtomicRmwOp() != RMWOp::FADD ||
      candidate.getScope() != MemSyncScope::GPU)
    return false;
  switch (candidate.getSem()) {
  case MemSemantic::ACQUIRE:
  case MemSemantic::RELEASE:
  case MemSemantic::ACQUIRE_RELEASE:
    break;
  default:
    // In particular, an already-relaxed atomic is not a new specialization.
    return false;
  }
  auto type = dyn_cast<RankedTensorType>(candidate.getType());
  if (!type || type.getRank() != 1 || !type.getElementType().isF32() ||
      type.getDimSize(0) <= 0)
    return false;
  auto destination = pointerRoot(candidate.getPtr());
  if (!destination ||
      cast<PointerType>(destination.getType()).getAddressSpace() !=
          PtrAddrSpace::Global ||
      !argumentFlag(destination, "tt.experimental_private") ||
      !argumentFlag(destination, "tt.experimental_num_ctas") ||
      !closedPrivateEffects(function, candidate, destination))
    return false;

  // The only write is this reduction, its old value is unobserved, and every
  // ordinary read is disjoint from its destination. Thus other programs of
  // this kernel cannot use it to publish a write or acquire an observable
  // memory effect. The private-output contract excludes external access until
  // invocation completion; no caller assertion about the absence of internal
  // synchronization is needed. Acquire/release ordering therefore adds no
  // observable behavior in this bounded function.
  //
  // Unlike specialize(), this proof needs no destination injectivity: the
  // atomic remains atomic, so colliding indices still reduce correctly. It
  // preserves the allowed nondeterministic floating-point serialization rather
  // than promising one fixed summation order. No address, value or mask
  // changes.
  candidate.setSem(MemSemantic::RELAXED);
  function->setAttr(
      "tt.experimental_private_atomic_relaxed",
      IntegerAttr::get(IntegerType::get(function.getContext(), 32), 1));
  return true;
}

bool specialize(FuncOp function) {
  if (!isCompleteKernelBody(function))
    return false;
  AtomicRMWOp candidate;
  for (Operation &op : function.getBody().front()) {
    if (auto atomic = dyn_cast<AtomicRMWOp>(op)) {
      if (candidate)
        return false;
      candidate = atomic;
    }
  }
  if (!candidate || !candidate.getResult().use_empty() ||
      candidate.getAtomicRmwOp() != RMWOp::FADD ||
      candidate.getScope() != MemSyncScope::GPU)
    return false;
  auto type = dyn_cast<RankedTensorType>(candidate.getType());
  if (!type || type.getRank() != 1 || !type.getElementType().isF32() ||
      type.getDimSize(0) <= 0)
    return false;
  auto destination = pointerRoot(candidate.getPtr());
  if (!destination ||
      cast<PointerType>(destination.getType()).getAddressSpace() !=
          PtrAddrSpace::Global ||
      !argumentFlag(destination, "tt.experimental_private") ||
      !argumentFlag(destination, "tt.experimental_num_ctas"))
    return false;
  auto gridX = argumentInteger(destination, "tt.experimental_grid_x");
  auto gridY = argumentInteger(destination, "tt.experimental_grid_y");
  auto gridZ = argumentInteger(destination, "tt.experimental_grid_z");
  if (!gridX || !gridY || !gridZ || *gridX <= 0 || *gridY <= 0 || *gridZ != 1 ||
      !closedPrivateEffects(function, candidate, destination))
    return false;
  IndexedAddressProof proof(*gridX, *gridY, type.getDimSize(0));
  if (!proof.prove(candidate.getPtr(), candidate.getMask(), destination))
    return false;

  OpBuilder builder(candidate);
  Location loc = candidate.getLoc();
  Value zero =
      arith::ConstantOp::create(builder, loc, type, builder.getZeroAttr(type));
  Value old = zero;
  if (!argumentFlag(destination, "tt.experimental_initial_zero")) {
    old = candidate.getMask()
              ? LoadOp::create(builder, loc, candidate.getPtr(),
                               candidate.getMask(), zero)
              : LoadOp::create(builder, loc, candidate.getPtr());
  }
  // CUDA global f32 atomicAdd flushes subnormal inputs/results. Ordinary arith
  // addf would change this numerical behavior, and +0 must not fold into a
  // copy.
  auto updated = ElementwiseInlineAsmOp::create(
      builder, loc, TypeRange{type}, "add.rn.ftz.f32 $0, $1, $2;", "=f,f,f",
      true, 1, ValueRange{old, candidate.getVal()});
  StoreOp::create(builder, loc, candidate.getPtr(), updated.getResult().front(),
                  candidate.getMask());
  candidate.erase();
  function->setAttr("tt.experimental_indexed_update_specialized",
                    builder.getI32IntegerAttr(1));
  return true;
}

} // namespace

class SpecializeIndexedUpdatesPass
    : public impl::TritonSpecializeIndexedUpdatesBase<
          SpecializeIndexedUpdatesPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    if (target != "cuda")
      return;
    ModuleOp module = getOperation();
    if (Attribute attr = module->getAttr("ttg.num-ctas")) {
      auto numCTAs = dyn_cast<IntegerAttr>(attr);
      if (!numCTAs || !numCTAs.getValue().isOne())
        return;
    }
    for (FuncOp function : module.getOps<FuncOp>())
      specialize(function);
  }
};

class RelaxPrivateAtomicsPass
    : public impl::TritonRelaxPrivateAtomicsBase<RelaxPrivateAtomicsPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    if (target != "cuda")
      return;
    ModuleOp module = getOperation();
    if (Attribute attr = module->getAttr("ttg.num-ctas")) {
      auto numCTAs = dyn_cast<IntegerAttr>(attr);
      if (!numCTAs || !numCTAs.getValue().isOne())
        return;
    }
    for (FuncOp function : module.getOps<FuncOp>())
      relaxPrivateAtomic(function);
  }
};

} // namespace mlir::triton
