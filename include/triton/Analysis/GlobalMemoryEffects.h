#ifndef TRITON_ANALYSIS_GLOBAL_MEMORY_EFFECTS_H
#define TRITON_ANALYSIS_GLOBAL_MEMORY_EFFECTS_H

#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>

namespace mlir {
class Operation;

namespace triton {
class FuncOp;

/// Contract-aware effects for ordinary pointer loads and stores in one
/// function.
///
/// Different nonnegative tt.experimental_alias_group values on entry arguments
/// promise disjoint actual access footprints, including all derived accesses.
/// Equal, missing or malformed groups may alias. Groups from distinct functions
/// cannot be compared. This is not a distinct-base-address assumption.
///
/// Explicit immutable contents are a separate caller promise: initialized at
/// entry and unchanged through every alias for the invocation. Neither alias
/// separation nor a no-modification result establishes initialization, absence
/// of observers, safe speculation, address invariance or synchronization
/// removal. Consumers must prove those obligations as required by their
/// transformation.
///
/// The summary accepts ordinary loads/stores and recursively inspected effects.
/// Calls, atomics, barriers, volatile accesses, asynchronous memory effects,
/// unknown pointer provenance and unknown effects block mayModify proofs.
/// Recursive-effect containers are inspected through their children; this is
/// not a proof that their execution regions may be merged or rescheduled.
/// Shared-memory descriptors and tensor-memory allocations are not modeled.
///
/// This object snapshots writes and unknown effects when constructed. Rebuild
/// it after changes to argument contracts, pointer provenance, memory effects
/// or function structure. Moving ordinary loads without changing their
/// provenance or removing supported nonvolatile loads preserves its
/// conservative summary.
class GlobalMemoryEffects {
public:
  explicit GlobalMemoryEffects(FuncOp function);

  /// Resolve a pointer to an entry argument of the analyzed function. Follow
  /// pointer representation operations and bounded scf.for recurrences only
  /// when each backedge derives from its exact carried pointer. A stable root
  /// does not imply a stable address. Unknown and cross-function roots fail.
  BlockArgument getArgumentRoot(Value pointer) const;

  std::optional<int64_t> getAliasGroup(Value pointer) const;
  bool isExplicitlyImmutable(Value pointer) const;
  bool isExplicitlyImmutableGlobal(Value pointer) const;

  /// May the two argument-derived access footprints overlap?
  bool mayAlias(Value first, Value second) const;

  /// May any operation in the analyzed function change the queried footprint?
  /// This conservative effect query does not use an immutable contract to
  /// disregard writes. It returns true for an unknown root or unknown effects.
  bool mayModify(Value pointer) const;

  /// Check a scope's operations recursively without making alias assumptions.
  /// This does not establish no-clobber, execution or dominance facts. Ordinary
  /// stores are allowed regardless of their destination. When the
  /// consumer has separately established initialized immutable GLOBAL contents,
  /// it may permit existing CTA barriers without moving/removing those
  /// barriers. Other synchronization and unknown effects remain unsupported.
  static bool hasOnlyOrdinaryEffects(Operation *scope,
                                     bool allowImmutableCTABarriers = false);

private:
  Operation *function;
  SmallVector<BlockArgument> writtenArguments;
  bool hasUnknownEffects = false;
};

} // namespace triton
} // namespace mlir

#endif // TRITON_ANALYSIS_GLOBAL_MEMORY_EFFECTS_H
