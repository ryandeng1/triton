#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/MathExtras.h"

#include <array>
#include <limits>

namespace mlir::triton {
#define GEN_PASS_DEF_TRITONPRIVATEREDUCTIONOWNERSHIP
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

std::optional<int64_t> argumentFact(BlockArgument arg, StringRef name) {
  auto fn = dyn_cast<FuncOp>(arg.getOwner()->getParentOp());
  if (!fn || arg.getOwner() != &fn.getBody().front())
    return std::nullopt;
  auto attr = fn.getArgAttrOfType<IntegerAttr>(arg.getArgNumber(), name);
  if (!attr || !attr.getType().isInteger(32))
    return std::nullopt;
  return attr.getInt();
}

std::optional<int64_t> knownInteger(Value value) {
  if (!value)
    return std::nullopt;
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    if (arg.getType().isInteger(32))
      return argumentFact(arg, "tt.experimental_value");
    return std::nullopt;
  }
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
  return integer.getBitWidth() == 1 ? integer.getZExtValue()
                                    : integer.getSExtValue();
}

bool flag(BlockArgument arg, StringRef name) {
  return argumentFact(arg, name) == 1;
}

bool representation(Operation *op) {
  return isa<SplatOp, BroadcastOp, ExpandDimsOp, ReshapeOp>(op);
}

BlockArgument root(Value pointer) {
  while (auto *op = pointer.getDefiningOp()) {
    if (!isa<AddPtrOp>(op) && !representation(op))
      return {};
    pointer = op->getOperand(0);
  }
  auto arg = dyn_cast<BlockArgument>(pointer);
  if (!arg || !isa<PointerType>(arg.getType()))
    return {};
  auto fn = dyn_cast<FuncOp>(arg.getOwner()->getParentOp());
  return fn && arg.getOwner() == &fn.getBody().front() ? arg : BlockArgument{};
}

bool global(BlockArgument arg) {
  return arg && cast<PointerType>(arg.getType()).getAddressSpace() ==
                    PtrAddrSpace::Global;
}

struct Affine {
  enum Term { ROW, GROUP, LANE, CONSTANT, SIZE };
  std::array<int64_t, SIZE> c{};
  bool operator==(const Affine &) const = default;
  bool constant() const { return c[ROW] == 0 && c[GROUP] == 0 && c[LANE] == 0; }
};

class AddressProof {
public:
  AddressProof(Value row, Value group, Value lane, int64_t rows, int64_t groups,
               int64_t block)
      : row(row), group(group), lane(lane),
        maxima{rows - 1, std::min(rows, groups) - 1, block - 1, 1} {}

  std::optional<Affine> integer(Value value) {
    if (!getElementTypeOrSelf(value.getType()).isInteger(32))
      return std::nullopt;
    Affine result;
    if (value == row) {
      result.c[Affine::ROW] = 1;
    } else if (value == group) {
      result.c[Affine::GROUP] = 1;
    } else if (value == lane) {
      result.c[Affine::LANE] = 1;
    } else if (auto constant = knownInteger(value)) {
      if (*constant < 0)
        return std::nullopt;
      result.c[Affine::CONSTANT] = *constant;
    } else if (Operation *op = value.getDefiningOp()) {
      if (representation(op))
        return integer(op->getOperand(0));
      if (auto add = dyn_cast<arith::AddIOp>(op))
        return bounded(sum(integer(add.getLhs()), integer(add.getRhs())));
      if (auto mul = dyn_cast<arith::MulIOp>(op)) {
        auto left = integer(mul.getLhs()), right = integer(mul.getRhs());
        if (!left || !right)
          return std::nullopt;
        if (left->constant())
          return bounded(scale(*right, left->c[Affine::CONSTANT]));
        if (right->constant())
          return bounded(scale(*left, right->c[Affine::CONSTANT]));
        return std::nullopt;
      }
      return std::nullopt;
    } else {
      return std::nullopt;
    }
    return bounded(result);
  }

  std::optional<Affine> pointer(Value value, BlockArgument expected) {
    if (value == expected)
      return Affine{};
    Operation *op = value.getDefiningOp();
    if (!op)
      return std::nullopt;
    if (representation(op))
      return pointer(op->getOperand(0), expected);
    if (auto add = dyn_cast<AddPtrOp>(op))
      return bounded(
          sum(pointer(add.getPtr(), expected), integer(add.getOffset())));
    return std::nullopt;
  }

  bool columnMask(Value mask, int64_t columns, int64_t block) {
    if (!mask || knownInteger(mask) == 1)
      return columns == block;
    auto cmp = mask.getDefiningOp<arith::CmpIOp>();
    if (!cmp || (cmp.getPredicate() != arith::CmpIPredicate::slt &&
                 cmp.getPredicate() != arith::CmpIPredicate::ult))
      return false;
    Affine expected;
    expected.c[Affine::LANE] = 1;
    auto lhs = integer(cmp.getLhs()), rhs = integer(cmp.getRhs());
    return lhs && rhs && *lhs == expected && rhs->constant() &&
           rhs->c[Affine::CONSTANT] == columns && columns > 0 &&
           columns <= block;
  }

private:
  // Bound every intermediate expression, including padded lanes, before it
  // becomes an AddPtr offset. Independent i32 offsets are sign-extended before
  // pointer addition; cancellation in a flattened final expression would not
  // justify ignoring intermediate overflow. Negative terms are unsupported.
  std::optional<Affine> bounded(std::optional<Affine> value) const {
    if (!value)
      return std::nullopt;
    int64_t maximum = 0;
    for (unsigned i = 0; i < Affine::SIZE; ++i) {
      int64_t term;
      if (value->c[i] < 0 || llvm::MulOverflow(value->c[i], maxima[i], term) ||
          llvm::AddOverflow(maximum, term, maximum) ||
          maximum > std::numeric_limits<int32_t>::max())
        return std::nullopt;
    }
    return value;
  }

  static std::optional<Affine> sum(std::optional<Affine> left,
                                   std::optional<Affine> right) {
    if (!left || !right)
      return std::nullopt;
    for (unsigned i = 0; i < Affine::SIZE; ++i)
      if (llvm::AddOverflow(left->c[i], right->c[i], left->c[i]))
        return std::nullopt;
    return left;
  }

  static std::optional<Affine> scale(Affine value, int64_t factor) {
    if (factor < 0)
      return std::nullopt;
    for (auto &coefficient : value.c)
      if (llvm::MulOverflow(coefficient, factor, coefficient))
        return std::nullopt;
    return value;
  }

  Value row, group, lane;
  std::array<int64_t, Affine::SIZE> maxima;
};

bool equalConstant(arith::CmpIOp cmp, Value value, int64_t constant) {
  return cmp && cmp.getPredicate() == arith::CmpIPredicate::eq &&
         ((cmp.getLhs() == value && knownInteger(cmp.getRhs()) == constant) ||
          (cmp.getRhs() == value && knownInteger(cmp.getLhs()) == constant));
}

bool exchange(AtomicRMWOp op, Value pointer, int64_t value) {
  return op && op.getAtomicRmwOp() == RMWOp::XCHG && op.getPtr() == pointer &&
         knownInteger(op.getVal()) == value &&
         (!op.getMask() || knownInteger(op.getMask()) == 1) &&
         op.getType().isInteger(32) && op.getResult().use_empty() &&
         op.getSem() == MemSemantic::ACQUIRE_RELEASE &&
         op.getScope() == MemSyncScope::GPU;
}

// Restrict computation to ordinary arithmetic and the logical block operations
// needed for a row calculation. Arbitrary pure inline assembly is not accepted.
bool pureLeaf(Operation *op) {
  if (op->getNumRegions() || !isMemoryEffectFree(op))
    return false;
  StringRef dialect = op->getName().getDialectNamespace();
  return dialect == "arith" || dialect == "math" || representation(op) ||
         isa<AddPtrOp, MakeRangeOp, GetProgramIdOp, ReduceReturnOp>(op);
}

bool pureReduction(ReduceOp reduce) {
  return !reduce
              ->walk([&](Operation *op) {
                StringRef dialect = op->getName().getDialectNamespace();
                bool pureCombiner = !op->getNumRegions() &&
                                    isMemoryEffectFree(op) &&
                                    (dialect == "arith" || dialect == "math" ||
                                     isa<ReduceReturnOp>(op));
                return op == reduce.getOperation() || pureCombiner
                           ? WalkResult::advance()
                           : WalkResult::interrupt();
              })
              .wasInterrupted();
}

struct Protocol {
  FuncOp function;
  GetProgramIdOp pid;
  MakeRangeOp lane;
  scf::WhileOp spin;
  AtomicCASOp acquire;
  LoadOp count;
  scf::IfOp first;
  std::array<StoreOp, 2> stores;
  std::array<arith::AddFOp, 2> additions;
  std::array<Value, 2> contributions;
  StoreOp rowStore;
  SmallVector<Operation *> prefix;
  BlockArgument locks, rowOutput;
  std::array<BlockArgument, 2> partials;
  Value group;
  int64_t groups = 0, rows = 0, columns = 0, stride = 0;

  bool match(FuncOp fn) {
    function = fn;
    if (!fn.isPublic() || fn.getNumResults() || fn.getBody().empty() ||
        !llvm::hasSingleElement(fn.getBody()))
      return false;
    auto uses = SymbolTable::getSymbolUses(fn, fn->getParentOp());
    if (!uses || !uses->empty())
      return false;
    SmallVector<Operation *> tail;
    for (Operation &op : fn.getBody().front()) {
      if (!spin) {
        if (auto loop = dyn_cast<scf::WhileOp>(op)) {
          spin = loop;
          continue;
        }
        prefix.push_back(&op);
        if (auto id = dyn_cast<GetProgramIdOp>(op)) {
          if (pid || id.getAxis() != ProgramIDDim::X)
            return false;
          pid = id;
        }
        if (auto range = dyn_cast<MakeRangeOp>(op)) {
          if (lane || range.getStart() != 0 || range.getEnd() <= 0)
            return false;
          lane = range;
        }
      } else {
        tail.push_back(&op);
      }
    }
    if (!pid || !lane || !spin || tail.size() != 8 || !isa<ReturnOp>(tail[7]) ||
        tail[7]->getNumOperands())
      return false;
    count = dyn_cast<LoadOp>(tail[0]);
    auto zeroTest = dyn_cast<arith::CmpIOp>(tail[1]);
    first = dyn_cast<scf::IfOp>(tail[2]);
    stores = {dyn_cast<StoreOp>(tail[3]), dyn_cast<StoreOp>(tail[4])};
    auto barrier = dyn_cast<gpu::BarrierOp>(tail[5]);
    auto release = dyn_cast<AtomicRMWOp>(tail[6]);
    if (!count || count.getIsVolatile() || count.getMask() ||
        !count.getResult().hasOneUse() ||
        !equalConstant(zeroTest, count.getResult(), 0) ||
        !zeroTest.getResult().hasOneUse() || !first ||
        first.getCondition() != zeroTest.getResult() ||
        first.getNumResults() != 2 || !stores[0] || !stores[1] || !barrier ||
        barrier.isWarp() || barrier.getAddrSpace() != gpu::AddrSpace::All)
      return false;
    if (spin.getNumOperands() || spin.getNumResults() ||
        !llvm::hasSingleElement(spin.getBefore()) ||
        !llvm::hasSingleElement(spin.getAfter()))
      return false;
    auto &before = spin.getBefore().front();
    auto &after = spin.getAfter().front();
    if (before.getNumArguments() || after.getNumArguments() ||
        before.getOperations().size() != 3 ||
        after.getOperations().size() != 1 ||
        !isa<scf::YieldOp>(after.front()) || after.front().getNumOperands())
      return false;
    acquire = dyn_cast<AtomicCASOp>(&before.front());
    auto test = dyn_cast<arith::CmpIOp>(before.front().getNextNode());
    auto condition = dyn_cast<scf::ConditionOp>(&before.back());
    if (!acquire || !acquire.getType().isInteger(32) ||
        acquire.getSem() != MemSemantic::ACQUIRE_RELEASE ||
        acquire.getScope() != MemSyncScope::GPU ||
        knownInteger(acquire.getCmp()) != 0 ||
        knownInteger(acquire.getVal()) != 1 ||
        !acquire.getResult().hasOneUse() || !equalConstant(test, acquire, 1) ||
        !test.getResult().hasOneUse() || !condition ||
        condition.getCondition() != test.getResult() ||
        condition.getArgs().size() || !exchange(release, acquire.getPtr(), 0))
      return false;
    auto lockAddress = acquire.getPtr().getDefiningOp<AddPtrOp>();
    if (!lockAddress)
      return false;
    locks = dyn_cast<BlockArgument>(lockAddress.getPtr());
    group = lockAddress.getOffset();
    Value dividend, divisor;
    if (auto rem = group.getDefiningOp<arith::RemSIOp>()) {
      dividend = rem.getLhs();
      divisor = rem.getRhs();
    } else if (auto rem = group.getDefiningOp<arith::RemUIOp>()) {
      dividend = rem.getLhs();
      divisor = rem.getRhs();
    }
    auto size = divisor ? knownInteger(divisor) : std::nullopt;
    if (!global(locks) ||
        !cast<PointerType>(locks.getType()).getPointeeType().isInteger(32) ||
        dividend != pid.getResult() || !size || *size <= 0)
      return false;
    groups = *size;
    if (!flag(locks, "tt.experimental_private") ||
        !flag(locks, "tt.experimental_initial_zero") ||
        !flag(locks, "tt.experimental_num_ctas") ||
        argumentFact(locks, "tt.experimental_grid_y") != 1 ||
        argumentFact(locks, "tt.experimental_grid_z") != 1)
      return false;
    auto grid = argumentFact(locks, "tt.experimental_grid_x");
    if (!grid || *grid <= 0)
      return false;
    rows = *grid;
    if (!llvm::hasSingleElement(first.getThenRegion()) ||
        !llvm::hasSingleElement(first.getElseRegion()))
      return false;
    auto &thenBlock = first.getThenRegion().front();
    auto &elseBlock = first.getElseRegion().front();
    if (thenBlock.getNumArguments() || elseBlock.getNumArguments() ||
        thenBlock.getOperations().size() != 2 ||
        elseBlock.getOperations().size() != 5 ||
        !exchange(dyn_cast<AtomicRMWOp>(thenBlock.front()), count.getPtr(), 1))
      return false;
    auto thenYield = dyn_cast<scf::YieldOp>(thenBlock.getTerminator());
    auto elseYield = dyn_cast<scf::YieldOp>(elseBlock.getTerminator());
    if (!thenYield || !elseYield || thenYield.getNumOperands() != 2 ||
        elseYield.getNumOperands() != 2)
      return false;
    llvm::SmallPtrSet<Operation *, 8> elseOperations;
    elseOperations.insert(elseYield.getOperation());
    AddressProof proof(pid, group, lane, rows, groups, lane.getEnd());
    Affine countOffset;
    countOffset.c[Affine::GROUP] = 1;
    countOffset.c[Affine::CONSTANT] = groups;
    auto actualCount = proof.pointer(count.getPtr(), locks);
    if (!actualCount || *actualCount != countOffset)
      return false;
    for (unsigned i = 0; i < 2; ++i) {
      auto type = dyn_cast<RankedTensorType>(first.getResult(i).getType());
      if (!type || type.getRank() != 1 || !type.getElementType().isF32() ||
          type.getDimSize(0) != lane.getEnd() ||
          stores[i].getValue() != first.getResult(i) ||
          !first.getResult(i).hasOneUse())
        return false;
      contributions[i] = thenYield.getOperand(i);
      if (contributions[i].getType() != type ||
          (contributions[i].getDefiningOp() &&
           contributions[i].getDefiningOp()->getParentOp() != fn))
        return false;
      additions[i] = elseYield.getOperand(i).getDefiningOp<arith::AddFOp>();
      if (!additions[i] || additions[i].getLhs() != contributions[i] ||
          !additions[i].getResult().hasOneUse())
        return false;
      auto old = additions[i].getRhs().getDefiningOp<LoadOp>();
      if (!old || old.getIsVolatile() || !old.getResult().hasOneUse() ||
          old.getPtr() != stores[i].getPtr() ||
          old.getMask() != stores[i].getMask())
        return false;
      elseOperations.insert(old.getOperation());
      elseOperations.insert(additions[i].getOperation());
      partials[i] = root(stores[i].getPtr());
      if (!global(partials[i]) ||
          !cast<PointerType>(partials[i].getType()).getPointeeType().isF32())
        return false;
      auto address = proof.pointer(stores[i].getPtr(), partials[i]);
      if (!address || address->c[Affine::ROW] ||
          address->c[Affine::LANE] != 1 || address->c[Affine::CONSTANT] ||
          address->c[Affine::GROUP] <= 0)
        return false;
      if (i == 0)
        columns = address->c[Affine::GROUP];
      if (address->c[Affine::GROUP] != columns ||
          !proof.columnMask(stores[i].getMask(), columns, lane.getEnd()) ||
          stores[i].getMask() != stores[0].getMask())
        return false;
    }
    if (elseOperations.size() != elseBlock.getOperations().size() ||
        !llvm::all_of(elseBlock, [&](Operation &op) {
          return elseOperations.contains(&op);
        }))
      return false;
    SmallVector<BlockArgument> reads;
    for (Operation *op : prefix) {
      if (auto load = dyn_cast<LoadOp>(op)) {
        auto arg = root(load.getPtr());
        if (load.getIsVolatile() || !global(arg))
          return false;
        reads.push_back(arg);
      } else if (auto store = dyn_cast<StoreOp>(op)) {
        if (rowStore)
          return false;
        rowStore = store;
      } else if (auto reduction = dyn_cast<ReduceOp>(op)) {
        if (!pureReduction(reduction))
          return false;
      } else if (!pureLeaf(op)) {
        return false;
      }
    }
    if (!rowStore || rowStore.getMask() != stores[0].getMask())
      return false;
    rowOutput = root(rowStore.getPtr());
    auto address = proof.pointer(rowStore.getPtr(), rowOutput);
    if (!global(rowOutput) || !address || address->c[Affine::GROUP] ||
        address->c[Affine::LANE] != 1 || address->c[Affine::CONSTANT] ||
        address->c[Affine::ROW] < columns)
      return false;
    stride = address->c[Affine::ROW];
    SmallVector<int64_t> outputGroups;
    for (auto output : {locks, rowOutput, partials[0], partials[1]}) {
      auto alias = argumentFact(output, "tt.experimental_alias_group");
      if (!flag(output, "tt.experimental_private") || !alias || *alias < 0 ||
          llvm::is_contained(outputGroups, *alias))
        return false;
      outputGroups.push_back(*alias);
    }
    for (auto input : reads) {
      auto alias = argumentFact(input, "tt.experimental_alias_group");
      if (!alias || *alias < 0 || llvm::is_contained(outputGroups, *alias))
        return false;
    }
    // All introduced induction arithmetic and original row/group addresses fit
    // signed i32. Inputs are evaluated only at original legal row IDs.
    constexpr int64_t limit = std::numeric_limits<int32_t>::max();
    int64_t lastRow, scratch;
    return rows - 1 <= limit - groups && groups <= (limit + 1) / 2 &&
           !llvm::MulOverflow(rows - 1, stride, lastRow) &&
           lastRow <= limit - (columns - 1) &&
           !llvm::MulOverflow(groups, columns, scratch) && scratch <= limit;
  }

  void rewrite() {
    Block *original = &function.getBody().front();
    auto *replacement = new Block();
    function.getBody().push_back(replacement);
    IRMapping arguments;
    for (auto arg : original->getArguments())
      arguments.map(arg, replacement->addArgument(arg.getType(), arg.getLoc()));
    OpBuilder builder(replacement, replacement->begin());
    Location loc = function.getLoc();
    Value owner = GetProgramIdOp::create(builder, loc, ProgramIDDim::X);
    Value extent = GetNumProgramsOp::create(builder, loc, ProgramIDDim::X);
    Value groupSize = arith::ConstantIntOp::create(builder, loc, groups, 32);
    Value active = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::slt, owner, groupSize);
    auto region = scf::IfOp::create(builder, loc, TypeRange{}, active, false);
    builder.setInsertionPointToStart(region.thenBlock());
    auto cloneRow = [&](Value row) {
      IRMapping mapping = arguments;
      mapping.map(pid.getResult(), row);
      for (Operation *op : prefix)
        if (op != pid.getOperation())
          builder.clone(*op, mapping);
      return mapping;
    };
    // Peel the first row: its contribution initializes the partial directly,
    // preserving signed zero and avoiding an extra floating addition by zero.
    auto firstRow = cloneRow(owner);
    SmallVector<Value> initial = {firstRow.lookup(contributions[0]),
                                  firstRow.lookup(contributions[1])};
    Value start = arith::AddIOp::create(builder, loc, owner, groupSize);
    auto loop =
        scf::ForOp::create(builder, loc, start, extent, groupSize, initial);
    builder.setInsertionPointToStart(loop.getBody());
    auto nextRow = cloneRow(loop.getInductionVar());
    SmallVector<Value> updated;
    for (unsigned i = 0; i < 2; ++i) {
      auto add = arith::AddFOp::create(builder, additions[i].getLoc(),
                                       nextRow.lookup(contributions[i]),
                                       loop.getRegionIterArg(i));
      add->setAttrs(additions[i]->getAttrs());
      updated.push_back(add);
    }
    if (!loop.getBody()->empty() &&
        loop.getBody()->back().hasTrait<OpTrait::IsTerminator>())
      loop.getBody()->back().erase();
    builder.setInsertionPointToEnd(loop.getBody());
    scf::YieldOp::create(builder, loc, updated);
    builder.setInsertionPointAfter(loop);
    for (unsigned i = 0; i < 2; ++i) {
      firstRow.map(first.getResult(i), loop.getResult(i));
      builder.clone(*stores[i].getOperation(), firstRow);
    }
    // Preserve observable post-invocation bookkeeping. No participant observes
    // these stores during execution under the private-state contract.
    StoreOp::create(builder, loc, firstRow.lookup(count.getPtr()),
                    firstRow.lookup(acquire.getVal()));
    StoreOp::create(builder, loc, firstRow.lookup(acquire.getPtr()),
                    firstRow.lookup(acquire.getCmp()));
    builder.setInsertionPointAfter(region);
    ReturnOp::create(builder, loc);
    original->erase();
    function->setAttr("tt.experimental_private_reduction_owned",
                      builder.getI32IntegerAttr(1));
  }
};

} // namespace

class PrivateReductionOwnershipPass
    : public impl::TritonPrivateReductionOwnershipBase<
          PrivateReductionOwnershipPass> {
public:
  using Base::Base;
  void runOnOperation() override {
    if (target != "cuda")
      return;
    ModuleOp module = getOperation();
    if (Attribute attr = module->getAttr("ttg.num-ctas")) {
      auto count = dyn_cast<IntegerAttr>(attr);
      if (!count || !count.getValue().isOne())
        return;
    }
    for (FuncOp function : module.getOps<FuncOp>()) {
      Protocol protocol;
      if (protocol.match(function))
        protocol.rewrite();
    }
  }
};
} // namespace mlir::triton
