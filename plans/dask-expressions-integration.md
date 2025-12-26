# Plan: Dask Expressions Integration for xarray

## Overview

This plan describes how to integrate xarray with Dask's new expression-based computation system. The goal is to allow xarray's Dataset and DataArray to participate in Dask's expression optimization pipeline, enabling better performance when computing multiple xarray objects together or mixing xarray with dask arrays.

**Status:** Phase 1 Complete, map_blocks support added, joint compute fixed

**Implementation Notes (added during development):**

- Dask automatically uses `.expr` via `collections_to_expr()` - users pass the Dataset/DataArray, not the expression
- Array expressions need `simplify()` + `lower_completely()` to become executable
- Tuple operands (like `var_exprs`) require custom `_simplify_down()` AND `_lower()` since optimizer only recurses into direct Expr operands
- Feature detection must check that `dask.array.Array` actually has `.expr` (not just that classes exist)
- Requires `DASK_ARRAY__QUERY_PLANNING=true` config to enable array expressions in dask
- Must override `.fuse()` on xarray expression classes to propagate fusion to nested array expressions (base `Expr.fuse()` is a no-op)
- Must override `_lower()` on xarray expression classes to lower nested array expressions (otherwise high-level expressions like `Reshape` won't be lowered and will fail when building the task graph)

**Related resources in dask repository (`../dask/` - note: previously referenced as `../dask3/`):**

- Design doc: `designs/array-expr.md` - Core principles of expression system
- Base classes: `dask/_expr.py` - `Expr`, `_ExprSequence`, `FinalizeCompute`
- Array expressions: `dask/array/_array_expr/_expr.py` - `ArrayExpr`, `FinalizeComputeArray`
- Compute flow: `dask/base.py` - `compute()`, `collections_to_expr()`

---

## Background: How Dask Expressions Work

### The Problem with HighLevelGraphs

Currently, xarray uses Dask's `HighLevelGraph` system. When you call `dask.compute(dataset)`:

1. xarray's `Dataset.__dask_graph__()` merges variable graphs via `HighLevelGraph.merge()`
2. Dask treats this as an opaque blob - no expression-level optimization
3. Shared subexpressions between variables may not be detected
4. Cannot jointly optimize xarray objects with other dask collections

### The Expression System

Dask's new expression system (used by dask-dataframe and dask-array) represents computations as expression trees that are optimized before graph generation:

```
User Code → Expression Tree → Optimize → Task Graph → Execute
```

Key benefits:

- **Simplification**: Algebraic rewrites (e.g., `arr[5:10][2:3]` → `arr[7:8]`)
- **Lowering**: Convert logical operations to physical implementations
- **Fusion**: Combine chains of operations into single tasks
- **Shared subexpressions**: Detect and preserve common computations

### Core Expr Interface

From `dask/_expr.py`, the essential methods are:

```python
class Expr:
    _parameters: list[str] = []  # Names of operands
    _defaults: dict = {}  # Default values for optional params
    operands: list  # Actual operand values

    def dependencies(self) -> list[Expr]:
        """Return Expr operands (for tree traversal)."""
        return [op for op in self.operands if isinstance(op, Expr)]

    def _layer(self) -> dict:
        """Return task graph layer added by this expression."""
        raise NotImplementedError

    def __dask_keys__(self) -> list:
        """Return output keys for this expression."""
        return [(self._name, i) for i in range(self.npartitions)]

    def __dask_graph__(self) -> dict:
        """Traverse tree, collect all layers."""
        stack = [self]
        seen = set()
        layers = []
        while stack:
            expr = stack.pop()
            if expr._name in seen:
                continue
            seen.add(expr._name)
            layers.append(expr._layer())
            for dep in expr.dependencies():
                stack.append(dep)
        return toolz.merge(layers)

    @functools.cached_property
    def _name(self) -> str:
        """Unique identifier for this expression."""
        return f"{self._funcname}-{self.deterministic_token}"

    def finalize_compute(self) -> Expr:
        """Wrap expression for compute (handles result reconstruction)."""
        return self

    # Optional optimization hooks
    def _simplify_down(self) -> Expr | None: ...
    def _simplify_up(self, parent, dependents) -> Expr | None: ...
    def _lower(self) -> Expr | None: ...
```

### How \_ExprSequence Works

When computing multiple collections together:

```python
dask.compute(arr1, arr2, dataset)
```

Dask's `collections_to_expr()` function (in `dask/base.py:414`) does:

```python
def collections_to_expr(collections, optimize_graph=True):
    exprs = []
    for coll in collections:
        if hasattr(coll, "expr"):
            exprs.append(coll.expr)  # Get underlying expression
        else:
            # Fallback: wrap HighLevelGraph in HLGExpr
            exprs.append(HLGExpr.from_collection(coll))

    if len(exprs) > 1:
        return _ExprSequence(*exprs)
    return exprs[0]
```

`_ExprSequence` (in `dask/_expr.py:1198`) holds multiple expressions and:

- Merges their layers in `_layer()`
- Returns nested keys in `__dask_keys__()`: `[[expr1_keys], [expr2_keys], ...]`
- Applies finalization to each in `finalize_compute()`
- Groups by type for smart fusion in `fuse()`

**Key insight:** The reconstruction of original collections happens via the `repack` function built by `unpack_collections()`, which uses `__dask_postcompute__`. The expression system only deals with the computation, not reconstruction.

---

## Proposed Design: DatasetExpr and DataArrayExpr

### Architecture

```
Dataset                    DataArray
    |                          |
    v                          v
DatasetExpr              DataArrayExpr
    |                          |
    +--- var_exprs: [ArrayExpr, ArrayExpr, ...]
    |
    +--- coord_exprs: [ArrayExpr, ...]
```

When `dask.compute(dataset)` is called:

1. `collections_to_expr` calls `dataset.expr`
2. Returns `DatasetExpr` containing all variable expressions
3. Optimizer sees all expressions together
4. `finalize_compute()` wraps for reconstruction
5. Results are assembled back into a Dataset

### Implementation

#### File: `xarray/core/dask_expr.py`

```python
"""
Dask expression integration for xarray.

This module provides expression classes that allow xarray Dataset and DataArray
to participate in Dask's expression-based optimization pipeline.

Requires dask >= X.Y.Z with array expression support.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from xarray import DataArray, Dataset


# Check for expression support
def _has_expr_support() -> bool:
    """Check if dask has expression support."""
    try:
        from dask._expr import Expr
        from dask.array._array_expr._expr import ArrayExpr

        return True
    except ImportError:
        return False


HAS_EXPR_SUPPORT = _has_expr_support()


if HAS_EXPR_SUPPORT:
    import toolz
    from dask._expr import Expr
    from dask._task_spec import DataNode, Task, TaskRef, List as TaskList
    from dask.base import tokenize

    class DatasetExpr(Expr):
        """Expression representing an xarray Dataset with chunked variables.

        This expression holds references to all variable and coordinate
        expressions, allowing joint optimization when computing.

        Parameters
        ----------
        var_names : tuple[str, ...]
            Names of data variables (in order)
        var_exprs : tuple[Expr, ...]
            Expressions for each data variable
        coord_names : tuple[str, ...]
            Names of coordinates (in order)
        coord_exprs : tuple[Expr, ...]
            Expressions for each coordinate
        non_chunked_vars : dict
            Variables that are not chunked (numpy arrays, scalars)
        non_chunked_coords : dict
            Coordinates that are not chunked
        dims : dict
            Dimension sizes
        var_dims : dict
            Mapping of variable name to its dimension names
        attrs : dict | None
            Dataset attributes
        indexes : dict | None
            Dataset indexes (may need special handling)
        """

        _parameters = [
            "var_names",
            "var_exprs",
            "coord_names",
            "coord_exprs",
            "non_chunked_vars",
            "non_chunked_coords",
            "dims",
            "var_dims",
            "attrs",
            "indexes",
        ]
        _defaults = {
            "non_chunked_vars": {},
            "non_chunked_coords": {},
            "attrs": None,
            "indexes": None,
        }

        @functools.cached_property
        def _name(self) -> str:
            return f"dataset-{self.deterministic_token}"

        # Note: __dask_tokenize__ inherited from Expr - tokenizes all operands
        # Note: __dask_graph__ inherited from Expr - traverses dependencies(), collects _layer()

        def dependencies(self) -> list[Expr]:
            """Return all variable and coordinate expressions."""
            deps = []
            for expr in self.var_exprs:
                if isinstance(expr, Expr):
                    deps.append(expr)
            for expr in self.coord_exprs:
                if isinstance(expr, Expr):
                    deps.append(expr)
            return deps

        def _layer(self) -> dict:
            """Merge layers from all constituent expressions."""
            layers = {}
            for expr in self.var_exprs:
                if hasattr(expr, "_layer"):
                    layers.update(expr._layer())
            for expr in self.coord_exprs:
                if hasattr(expr, "_layer"):
                    layers.update(expr._layer())
            return layers

        def __dask_keys__(self) -> list:
            """Return keys for all chunked variables and coordinates.

            Structure: [[var1_keys...], [var2_keys...], [coord1_keys...], ...]
            """
            keys = []
            for expr in self.var_exprs:
                keys.append(list(expr.__dask_keys__()))
            for expr in self.coord_exprs:
                keys.append(list(expr.__dask_keys__()))
            return keys

        # Note: __dask_graph__ inherited from Expr - traverses dependencies(), merges _layer()

        def finalize_compute(self) -> DatasetExprFinalize:
            """Wrap for compute - handles result reconstruction."""
            return DatasetExprFinalize(
                var_names=self.var_names,
                var_exprs=tuple(
                    e.finalize_compute() if hasattr(e, "finalize_compute") else e
                    for e in self.var_exprs
                ),
                coord_names=self.coord_names,
                coord_exprs=tuple(
                    e.finalize_compute() if hasattr(e, "finalize_compute") else e
                    for e in self.coord_exprs
                ),
                non_chunked_vars=self.non_chunked_vars,
                non_chunked_coords=self.non_chunked_coords,
                dims=self.dims,
                var_dims=self.var_dims,
                attrs=self.attrs,
                indexes=self.indexes,
            )

        def __dask_annotations__(self) -> dict:
            """Merge annotations from all expressions."""
            annotations = {}
            for expr in self.dependencies():
                for k, v in expr.__dask_annotations__().items():
                    annotations.setdefault(k, {}).update(v)
            return annotations

        # Optional: Dataset-specific optimizations
        def _simplify_down(self):
            """Potential Dataset-level optimizations.

            Examples of what could be optimized:
            - If all variables have same slice applied, push down
            - Common subexpression detection across variables
            - Coordinate alignment optimizations
            """
            # Start with no optimizations, add later as needed
            return None

    class DatasetExprFinalize(Expr):
        """Handles reconstruction of Dataset after compute.

        This expression adds a final task that takes all computed arrays
        and reassembles them into an xarray Dataset.
        """

        _parameters = [
            "var_names",
            "var_exprs",
            "coord_names",
            "coord_exprs",
            "non_chunked_vars",
            "non_chunked_coords",
            "dims",
            "var_dims",
            "attrs",
            "indexes",
        ]
        _defaults = {
            "non_chunked_vars": {},
            "non_chunked_coords": {},
            "attrs": None,
            "indexes": None,
        }

        @functools.cached_property
        def _name(self) -> str:
            return f"dataset-finalize-{self.deterministic_token}"

        def dependencies(self) -> list[Expr]:
            deps = []
            for expr in self.var_exprs:
                if isinstance(expr, Expr):
                    deps.append(expr)
            for expr in self.coord_exprs:
                if isinstance(expr, Expr):
                    deps.append(expr)
            return deps

        def _layer(self) -> dict:
            """Build layer with reconstruction task."""
            # Collect keys from finalized expressions
            var_keys = []
            for expr in self.var_exprs:
                # After finalize_compute, arrays should have single key
                keys = list(expr.__dask_keys__())
                if len(keys) == 1 and not isinstance(keys[0], tuple):
                    var_keys.append(keys[0])
                else:
                    # Flatten nested keys
                    from dask.base import flatten

                    var_keys.extend(flatten(keys))

            coord_keys = []
            for expr in self.coord_exprs:
                keys = list(expr.__dask_keys__())
                if len(keys) == 1 and not isinstance(keys[0], tuple):
                    coord_keys.append(keys[0])
                else:
                    from dask.base import flatten

                    coord_keys.extend(flatten(keys))

            # Merge underlying layers
            layers = {}
            for expr in self.var_exprs:
                if hasattr(expr, "_layer"):
                    layers.update(expr._layer())
            for expr in self.coord_exprs:
                if hasattr(expr, "_layer"):
                    layers.update(expr._layer())

            # Add reconstruction task
            all_keys = var_keys + coord_keys
            layers[self._name] = Task(
                self._name,
                _reconstruct_dataset,
                TaskList(*[TaskRef(k) for k in all_keys]),
                # Pass metadata as data nodes (not task refs)
                DataNode(None, self.var_names),
                DataNode(None, self.coord_names),
                DataNode(None, self.non_chunked_vars),
                DataNode(None, self.non_chunked_coords),
                DataNode(None, self.var_dims),
                DataNode(None, self.attrs),
            )

            return layers

        def __dask_keys__(self) -> list:
            """Return single key for the reconstructed Dataset."""
            return [self._name]

        # Note: __dask_graph__ inherited from Expr - will include _layer() with reconstruction task

    def _reconstruct_dataset(
        computed_arrays: list,
        var_names: tuple[str, ...],
        coord_names: tuple[str, ...],
        non_chunked_vars: dict,
        non_chunked_coords: dict,
        var_dims: dict,
        attrs: dict | None,
    ):
        """Rebuild Dataset from computed numpy arrays.

        This function runs as the final task after all arrays are computed.
        """
        import xarray as xr

        n_vars = len(var_names)
        var_arrays = computed_arrays[:n_vars]
        coord_arrays = computed_arrays[n_vars:]

        # Build data_vars dict
        data_vars = {}
        for name, arr in zip(var_names, var_arrays):
            dims = var_dims.get(name, ())
            data_vars[name] = (dims, arr)

        # Add non-chunked variables
        for name, (dims, data) in non_chunked_vars.items():
            data_vars[name] = (dims, data)

        # Build coords dict
        coords = {}
        for name, arr in zip(coord_names, coord_arrays):
            dims = var_dims.get(name, (name,))  # Default: coord name is dim
            coords[name] = (dims, arr)

        # Add non-chunked coordinates
        for name, (dims, data) in non_chunked_coords.items():
            coords[name] = (dims, data)

        return xr.Dataset(data_vars, coords=coords, attrs=attrs)

    class DataArrayExpr(Expr):
        """Expression representing an xarray DataArray with chunked data.

        Simpler than DatasetExpr since there's only one data variable.
        """

        _parameters = [
            "name",
            "data_expr",
            "coord_names",
            "coord_exprs",
            "non_chunked_coords",
            "dims",
            "attrs",
        ]
        _defaults = {"name": None, "non_chunked_coords": {}, "attrs": None}

        @functools.cached_property
        def _name(self) -> str:
            return f"dataarray-{self.deterministic_token}"

        def dependencies(self) -> list[Expr]:
            deps = []
            if isinstance(self.data_expr, Expr):
                deps.append(self.data_expr)
            for expr in self.coord_exprs:
                if isinstance(expr, Expr):
                    deps.append(expr)
            return deps

        def _layer(self) -> dict:
            layers = {}
            if hasattr(self.data_expr, "_layer"):
                layers.update(self.data_expr._layer())
            for expr in self.coord_exprs:
                if hasattr(expr, "_layer"):
                    layers.update(expr._layer())
            return layers

        def __dask_keys__(self) -> list:
            keys = [list(self.data_expr.__dask_keys__())]
            for expr in self.coord_exprs:
                keys.append(list(expr.__dask_keys__()))
            return keys

        # Note: __dask_graph__ inherited from Expr

        def finalize_compute(self) -> DataArrayExprFinalize:
            return DataArrayExprFinalize(
                name=self.name,
                data_expr=(
                    self.data_expr.finalize_compute()
                    if hasattr(self.data_expr, "finalize_compute")
                    else self.data_expr
                ),
                coord_names=self.coord_names,
                coord_exprs=tuple(
                    e.finalize_compute() if hasattr(e, "finalize_compute") else e
                    for e in self.coord_exprs
                ),
                non_chunked_coords=self.non_chunked_coords,
                dims=self.dims,
                attrs=self.attrs,
            )

    class DataArrayExprFinalize(Expr):
        """Handles reconstruction of DataArray after compute."""

        _parameters = [
            "name",
            "data_expr",
            "coord_names",
            "coord_exprs",
            "non_chunked_coords",
            "dims",
            "attrs",
        ]
        _defaults = {"name": None, "non_chunked_coords": {}, "attrs": None}

        @functools.cached_property
        def _name(self) -> str:
            return f"dataarray-finalize-{self.deterministic_token}"

        def dependencies(self) -> list[Expr]:
            deps = []
            if isinstance(self.data_expr, Expr):
                deps.append(self.data_expr)
            for expr in self.coord_exprs:
                if isinstance(expr, Expr):
                    deps.append(expr)
            return deps

        def _layer(self) -> dict:
            # Get key for data
            data_keys = list(self.data_expr.__dask_keys__())
            if len(data_keys) == 1 and not isinstance(data_keys[0], tuple):
                data_key = data_keys[0]
            else:
                from dask.base import flatten

                data_key = list(flatten(data_keys))[
                    0
                ]  # Should be single after finalize

            coord_keys = []
            for expr in self.coord_exprs:
                keys = list(expr.__dask_keys__())
                if len(keys) == 1 and not isinstance(keys[0], tuple):
                    coord_keys.append(keys[0])
                else:
                    from dask.base import flatten

                    coord_keys.extend(flatten(keys))

            # Merge underlying layers
            layers = {}
            if hasattr(self.data_expr, "_layer"):
                layers.update(self.data_expr._layer())
            for expr in self.coord_exprs:
                if hasattr(expr, "_layer"):
                    layers.update(expr._layer())

            # Reconstruction task
            layers[self._name] = Task(
                self._name,
                _reconstruct_dataarray,
                TaskRef(data_key),
                TaskList(*[TaskRef(k) for k in coord_keys]),
                DataNode(None, self.name),
                DataNode(None, self.coord_names),
                DataNode(None, self.non_chunked_coords),
                DataNode(None, self.dims),
                DataNode(None, self.attrs),
            )

            return layers

        def __dask_keys__(self) -> list:
            return [self._name]

        # Note: __dask_graph__ inherited from Expr

    def _reconstruct_dataarray(
        data: Any,
        coord_arrays: list,
        name: str | None,
        coord_names: tuple[str, ...],
        non_chunked_coords: dict,
        dims: tuple[str, ...],
        attrs: dict | None,
    ):
        """Rebuild DataArray from computed numpy array."""
        import xarray as xr

        # Build coords dict
        coords = {}
        for cname, arr in zip(coord_names, coord_arrays):
            coords[cname] = arr

        # Add non-chunked coordinates
        for cname, (cdims, cdata) in non_chunked_coords.items():
            coords[cname] = (cdims, cdata) if cdims else cdata

        return xr.DataArray(data, dims=dims, coords=coords, name=name, attrs=attrs)


# Fallback for older dask versions
else:
    DatasetExpr = None
    DataArrayExpr = None
    DatasetExprFinalize = None
    DataArrayExprFinalize = None
```

#### Modifications to `xarray/core/dataset.py`

Add the `.expr` property:

```python
class Dataset:
    # ... existing code ...

    @property
    def expr(self):
        """Return expression for joint optimization with dask collections.

        This property returns a DatasetExpr that contains references to
        all chunked variable and coordinate expressions. When computed
        with other dask collections via `dask.compute()`, all expressions
        are optimized together, enabling detection of shared subexpressions
        and joint optimization.

        Returns
        -------
        DatasetExpr
            Expression representing this Dataset's computation.

        Raises
        ------
        ImportError
            If dask expression support is not available.
        ValueError
            If this Dataset has no chunked (dask-backed) variables.

        Examples
        --------
        >>> import dask
        >>> import dask.array as da
        >>> import xarray as xr
        >>>
        >>> # Create dataset with shared computation
        >>> base = da.random.random((1000, 1000), chunks=100)
        >>> ds = xr.Dataset({
        ...     'a': (['x', 'y'], base * 2),
        ...     'b': (['x', 'y'], base + 1),
        ... })
        >>>
        >>> # Compute with another dask array - all optimized together
        >>> result_ds, result_arr = dask.compute(ds, base.mean())
        """
        from xarray.core.dask_expr import DatasetExpr, HAS_EXPR_SUPPORT

        if not HAS_EXPR_SUPPORT:
            raise ImportError(
                "Dask expression support requires dask >= X.Y.Z. "
                "Install with: pip install 'dask[complete]>=X.Y.Z'"
            )

        # Collect chunked variables
        var_names = []
        var_exprs = []
        non_chunked_vars = {}

        # Collect chunked coordinates
        coord_names = []
        coord_exprs = []
        non_chunked_coords = {}

        # Track dimension info
        var_dims = {}

        for name, var in self._variables.items():
            is_coord = name in self._coord_names
            dims = var.dims
            var_dims[name] = dims

            # Check if variable is chunked
            if hasattr(var._data, "expr"):
                # Dask array with expression support
                expr = var._data.expr
                if is_coord:
                    coord_names.append(name)
                    coord_exprs.append(expr)
                else:
                    var_names.append(name)
                    var_exprs.append(expr)
            elif hasattr(var._data, "__dask_graph__"):
                # Dask array without expression (older dask?)
                # Could wrap in HLGExpr, or raise
                raise ValueError(
                    f"Variable '{name}' has dask array without expression support. "
                    "This may indicate an older dask version or custom array type."
                )
            else:
                # Non-chunked variable
                if is_coord:
                    non_chunked_coords[name] = (dims, var.values)
                else:
                    non_chunked_vars[name] = (dims, var.values)

        if not var_exprs and not coord_exprs:
            raise ValueError(
                "Dataset has no chunked variables. "
                "Use .expr only with dask-backed Datasets."
            )

        return DatasetExpr(
            var_names=tuple(var_names),
            var_exprs=tuple(var_exprs),
            coord_names=tuple(coord_names),
            coord_exprs=tuple(coord_exprs),
            non_chunked_vars=non_chunked_vars,
            non_chunked_coords=non_chunked_coords,
            dims=dict(self.dims),
            var_dims=var_dims,
            attrs=dict(self.attrs) if self.attrs else None,
            indexes=None,  # TODO: Handle indexes properly
        )
```

#### Modifications to `xarray/core/dataarray.py`

```python
class DataArray:
    # ... existing code ...

    @property
    def expr(self):
        """Return expression for joint optimization with dask collections.

        See Dataset.expr for full documentation.
        """
        from xarray.core.dask_expr import DataArrayExpr, HAS_EXPR_SUPPORT

        if not HAS_EXPR_SUPPORT:
            raise ImportError("Dask expression support requires dask >= X.Y.Z.")

        if not hasattr(self.variable._data, "expr"):
            if hasattr(self.variable._data, "__dask_graph__"):
                raise ValueError("DataArray has dask data without expression support.")
            raise ValueError(
                "DataArray is not chunked. Use .expr only with dask-backed DataArrays."
            )

        # Collect chunked coordinates
        coord_names = []
        coord_exprs = []
        non_chunked_coords = {}

        for name, coord in self.coords.items():
            if hasattr(coord.variable._data, "expr"):
                coord_names.append(name)
                coord_exprs.append(coord.variable._data.expr)
            else:
                dims = coord.dims
                non_chunked_coords[name] = (dims, coord.values)

        return DataArrayExpr(
            name=self.name,
            data_expr=self.variable._data.expr,
            coord_names=tuple(coord_names),
            coord_exprs=tuple(coord_exprs),
            non_chunked_coords=non_chunked_coords,
            dims=self.dims,
            attrs=dict(self.attrs) if self.attrs else None,
        )
```

---

## Implementation Phases

### Phase 1: Core Expression Classes ✓ COMPLETE

**Files created:**

- `xarray/core/dask_expr.py` - Expression classes
- `xarray/tests/test_dask_expr.py` - 17 tests

**Files modified:**

- `xarray/core/dataset.py` - Added `.expr` property (lines 634-736)
- `xarray/core/dataarray.py` - Added `.expr` property (lines 1105-1151)

**What was implemented:**

1. `DatasetExpr` / `DatasetExprFinalize` - container + reconstruction
2. `DataArrayExpr` / `DataArrayExprFinalize` - container + reconstruction
3. `HAS_EXPR_SUPPORT` - checks if dask.array has functional `.expr`
4. `.expr` properties on Dataset and DataArray
5. `_simplify_down()` to handle tuple operands and lower array expressions

**Key insight:** The integration is automatic - `dask.compute(ds)` uses `.expr` via `collections_to_expr()`. Users don't need to explicitly access `.expr`.

### Phase 2: Testing ✓ COMPLETE

**Test coverage (17 tests):**

- Basic compute for Dataset and DataArray
- Multiple variables
- Attributes preservation
- Non-chunked variables/coords
- Chunked coordinates
- Error handling for non-dask data
- Joint optimization with dask arrays
- Multiple datasets together
- Shared subexpression deduplication

**To run tests:** `DASK_ARRAY__QUERY_PLANNING=true PYTHONPATH=../dask pytest xarray/tests/test_dask_expr.py`

### Phase 3: Integration with Existing Dask Protocol

**Current status:** Using automatic integration via `.expr` property. Dask's `collections_to_expr()` checks for `.expr` attribute and uses it when present.

**Note:** The original plan suggested "Option A" (explicit `ds.expr`), but the actual dask implementation uses `.expr` automatically when the collection has it. Users just call `dask.compute(ds)` as before.

### Phase 4: Optimizations (Future)

Potential Dataset-specific optimizations in `_simplify_down`:

1. **Common slice pushdown**: If all variables have same slice, push to variables
2. **Coordinate alignment**: Optimize when variables share coordinates
3. **Broadcast optimization**: When variables broadcast together
4. **Groupby fusion**: Optimize groupby operations across variables

---

## Testing Strategy

### Unit Tests

```python
# xarray/tests/test_dask_expr.py

import pytest
import numpy as np

dask = pytest.importorskip("dask")
da = pytest.importorskip("dask.array")

from xarray.core.dask_expr import HAS_EXPR_SUPPORT

pytestmark = pytest.mark.skipif(
    not HAS_EXPR_SUPPORT, reason="Requires dask with expression support"
)


class TestDatasetExpr:
    def test_basic_compute(self):
        import xarray as xr

        data = da.ones((10, 10), chunks=5)
        ds = xr.Dataset({"a": (["x", "y"], data)})

        expr = ds.expr
        result = dask.compute(expr)[0]

        assert isinstance(result, xr.Dataset)
        assert "a" in result
        np.testing.assert_array_equal(result["a"].values, np.ones((10, 10)))

    def test_joint_optimization(self):
        import xarray as xr

        # Create shared base array
        base = da.random.random((100, 100), chunks=50)

        ds = xr.Dataset(
            {
                "a": (["x", "y"], base * 2),
                "b": (["x", "y"], base + 1),
            }
        )

        # Compute together - shared 'base' should be computed once
        result_ds, result_mean = dask.compute(ds.expr, base.mean())

        assert isinstance(result_ds, xr.Dataset)
        assert isinstance(result_mean, float)

    def test_preserves_attrs(self):
        import xarray as xr

        data = da.ones((10,), chunks=5)
        ds = xr.Dataset({"a": (["x"], data)}, attrs={"description": "test dataset"})

        result = dask.compute(ds.expr)[0]
        assert result.attrs["description"] == "test dataset"

    def test_non_chunked_variables(self):
        import xarray as xr

        chunked = da.ones((10,), chunks=5)
        non_chunked = np.array([1, 2, 3])

        ds = xr.Dataset(
            {
                "chunked": (["x"], chunked),
                "non_chunked": (["y"], non_chunked),
            }
        )

        result = dask.compute(ds.expr)[0]

        np.testing.assert_array_equal(result["chunked"].values, np.ones(10))
        np.testing.assert_array_equal(result["non_chunked"].values, [1, 2, 3])

    def test_coordinates_preserved(self):
        import xarray as xr

        data = da.ones((10, 10), chunks=5)
        x_coord = da.arange(10, chunks=5)
        y_coord = np.arange(10)  # Non-chunked coord

        ds = xr.Dataset({"a": (["x", "y"], data)}, coords={"x": x_coord, "y": y_coord})

        result = dask.compute(ds.expr)[0]

        np.testing.assert_array_equal(result.coords["x"].values, np.arange(10))
        np.testing.assert_array_equal(result.coords["y"].values, np.arange(10))

    def test_error_non_dask_dataset(self):
        import xarray as xr

        ds = xr.Dataset({"a": (["x"], np.ones(10))})

        with pytest.raises(ValueError, match="no chunked variables"):
            ds.expr


class TestDataArrayExpr:
    def test_basic_compute(self):
        import xarray as xr

        data = da.ones((10, 10), chunks=5)
        arr = xr.DataArray(data, dims=["x", "y"], name="test")

        result = dask.compute(arr.expr)[0]

        assert isinstance(result, xr.DataArray)
        assert result.name == "test"
        np.testing.assert_array_equal(result.values, np.ones((10, 10)))


class TestSharedSubexpressions:
    def test_shared_computation_not_duplicated(self):
        """Verify that shared computation between variables is computed once."""
        import xarray as xr

        # Track how many times the base computation runs
        call_count = [0]

        def expensive_op(x):
            call_count[0] += 1
            return x * 2

        # Create a base array with tracked computation
        base = da.ones((10, 10), chunks=5)
        base_computed = base.map_blocks(expensive_op, dtype=base.dtype)

        # Two variables that share the same base computation
        ds = xr.Dataset(
            {
                "a": (["x", "y"], base_computed + 1),
                "b": (["x", "y"], base_computed + 2),
            }
        )

        # Reset counter
        call_count[0] = 0

        # Compute via expression - should compute base_computed only once
        result = dask.compute(ds.expr)[0]

        # base_computed has 4 chunks (2x2), so expensive_op should run 4 times
        # NOT 8 times (which would happen if 'a' and 'b' each triggered it)
        assert (
            call_count[0] == 4
        ), f"Expected 4 calls, got {call_count[0]} - shared expr was duplicated!"

        # Verify results are correct
        expected = np.ones((10, 10)) * 2  # base after expensive_op
        np.testing.assert_array_equal(result["a"].values, expected + 1)
        np.testing.assert_array_equal(result["b"].values, expected + 2)

    def test_joint_optimization_with_dask_array(self):
        """Verify optimization when computing Dataset alongside raw dask arrays."""
        import xarray as xr

        call_count = [0]

        def tracked(x):
            call_count[0] += 1
            return x

        # Shared base
        base = da.ones((10, 10), chunks=5).map_blocks(tracked, dtype=float)

        # xarray dataset using base
        ds = xr.Dataset({"a": (["x", "y"], base + 1)})

        # Standalone dask array using same base
        arr = base.mean()

        call_count[0] = 0

        # Compute together via expressions
        result_ds, result_arr = dask.compute(ds.expr, arr)

        # base has 4 chunks, should only compute once total
        assert call_count[0] == 4, f"Expected 4, got {call_count[0]}"


class TestExpressionOptimizations:
    def test_slice_fused_into_creation(self):
        """Verify slice is fused into array creation, not a separate operation."""
        import xarray as xr
        from dask._expr import Expr

        # Create dataset with chunked array
        data = da.ones((100, 100), chunks=10)
        ds = xr.Dataset({"var": (["x", "y"], data)})

        # Transform through xarray
        ds = ds + 1
        ds = ds * 2

        # Slice through xarray
        ds = ds.isel(x=slice(5, 15))  # 100 -> 10 in first dim

        # Get expression for the variable
        expr = ds["var"].variable._data.expr

        def has_slice_expr(e, seen=None):
            """Check if tree contains a Slice expression."""
            if seen is None:
                seen = set()
            if e._name in seen:
                return False
            seen.add(e._name)

            if type(e).__name__ == "Slice":
                return True
            return any(has_slice_expr(dep, seen) for dep in e.dependencies())

        def get_leaf_shapes(e, seen=None):
            """Get shapes of leaf expressions (no dependencies)."""
            if seen is None:
                seen = set()
            if e._name in seen:
                return []
            seen.add(e._name)

            deps = e.dependencies()
            if not deps:
                # Leaf node - return its shape if it has one
                if hasattr(e, "shape"):
                    return [e.shape]
                return []

            shapes = []
            for dep in deps:
                shapes.extend(get_leaf_shapes(dep, seen))
            return shapes

        # Before optimization: should have Slice in tree
        assert has_slice_expr(expr), "Expected Slice in unoptimized expression"

        # Before optimization: leaf should have original shape
        leaf_shapes_before = get_leaf_shapes(expr)
        assert any(
            s == (100, 100) for s in leaf_shapes_before
        ), f"Expected (100, 100) leaf shape before optimization, got {leaf_shapes_before}"

        # After optimization
        optimized = expr.optimize()

        # Slice should be gone (fused into creation)
        assert not has_slice_expr(
            optimized
        ), "Slice still present after optimization - not fused into creation"

        # Leaf should have sliced shape
        leaf_shapes_after = get_leaf_shapes(optimized)
        assert any(
            s == (10, 100) for s in leaf_shapes_after
        ), f"Expected (10, 100) leaf shape after optimization, got {leaf_shapes_after}"

    def test_slice_pushdown_reduces_io(self):
        """Verify slice pushdown happens automatically and reduces I/O."""
        import xarray as xr
        import zarr
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/test.zarr"
            z = zarr.open(
                path, mode="w", shape=(1000, 1000), chunks=(100, 100), dtype="f8"
            )
            z[:] = np.random.random((1000, 1000))

            read_slices = []
            original_getitem = z.__class__.__getitem__

            def tracking_getitem(self, key):
                read_slices.append(key)
                return original_getitem(self, key)

            z.__class__.__getitem__ = tracking_getitem

            try:
                # Load via xarray
                ds = xr.open_zarr(path)

                # Transform through xarray operations
                ds = ds + 1
                ds = ds * 2

                # Slice through xarray
                ds = ds.isel(dim_0=slice(0, 50), dim_1=slice(0, 50))

                # Normal compute - no explicit optimize()
                read_slices.clear()
                result = dask.compute(ds.expr)[0]

                # Should only read first chunk, not all 100
                assert (
                    len(read_slices) <= 1
                ), f"Expected 1 chunk read (slice pushdown), got {len(read_slices)}"

                expected = (z[0:50, 0:50] + 1) * 2
                np.testing.assert_array_almost_equal(result["data"].values, expected)

            finally:
                z.__class__.__getitem__ = original_getitem
```

---

## Open Questions

1. **Index handling**: How should xarray indexes be preserved through expression compute? May need special handling or simplification.

2. **Encoding**: Should variable encoding be preserved? Currently not included.

3. **Chunked coordinates**: Are there edge cases with multi-dimensional coordinates?

4. **Memory**: Does holding references to expressions prevent garbage collection of intermediate results?

5. **Distributed**: Any special considerations for dask.distributed?

6. **Alternative backends**: How does this interact with other chunked array backends (cubed, etc.) via the ChunkManager abstraction?

---

## Known Issues (Discovered During Stress Testing)

The following issues were discovered during stress testing with complex xarray workflows. These represent gaps in the current integration that need to be addressed.

### Critical Issues

#### 1. `map_blocks` Broken - Relies on HighLevelGraph ✓ FIXED

```python
ds.map_blocks(my_func)
# Previously: AttributeError: 'Array' object has no attribute '__dask_layers__'
# Now: Works correctly with expression-based arrays
```

**Cause**: xarray's `map_blocks` built a `HighLevelGraph` directly, which required `__dask_layers__` on input arrays. Expression-based dask arrays don't have this attribute.

**Fix implemented**: Created new expression classes in `xarray/core/dask_expr.py`:

- `MapBlocksSharedExpr`: Generates the shared wrapper tasks that call the user function
- `MapBlocksVarExpr`: ArrayExpr that extracts individual output variables from wrapper results

Modified `xarray/core/parallel.py` to detect expression-based arrays via `_uses_expr_arrays()` and route to `_map_blocks_expr()` which builds the expression tree instead of HighLevelGraph.

**Key implementation details**:

- `MapBlocksSharedExpr` stores expressions in flat tuples (`input_var_exprs`, `input_coord_exprs`) separate from metadata (`input_var_meta`, `input_coord_meta`)
- This allows dask's optimizer to traverse the expressions automatically (dask walks tuples when `dependencies()` is overridden)
- No custom `_simplify_down()` or `_lower()` needed - the optimizer handles it
- `_layer()` uses `self.gname` for stable task keys
- Wrapper tasks are keyed by `gname`, extraction tasks reference `gname` for consistency

#### 2. Joint Compute Fails - `_ExprSequence` Missing xarray Attributes ✓ FIXED

```python
ds1 = xr.Dataset({"a": ...})
ds2 = xr.Dataset({"b": ...})
dask.compute(ds1, ds2)  # NOW WORKS
```

**Solution**: The `fuse()` methods in all xarray expression classes now detect when `self` is an `_ExprSequence` (dask calls `fuse.__func__(seq)` to pass the sequence as self). When this happens, `_fuse_xarray_sequence()` is called to:

1. Collect all array expressions from all operands in the sequence
2. Fuse them jointly using `_fuse_exprs()` to preserve shared subexpressions
3. Reconstruct each operand with its fused expressions (using `type(op)` to handle mixed Dataset/DataArray sequences)
4. Return a new `_ExprSequence` with the fused operands

**Known limitation**: When computing xarray objects alongside raw dask arrays (`dask.compute(ds, arr)`), they are fused separately because `_ExprSequence.fuse()` groups by module. Shared subexpressions between xarray and raw dask arrays are not deduplicated.

### Runtime Errors

#### 3. Rolling with Chunks Smaller Than Window

```python
# ERA5-like data with per-timestep chunks
data = da.random.random((100, 50, 100), chunks=(1, 50, 100))
ds = xr.Dataset({"temp": (["time", "lat", "lon"], data)})
ds.rolling(time=30, center=True).mean().compute()
# ValueError: Moving window (=30) must between 1 and 29, inclusive
```

**Cause**: When chunk size along rolling dimension is smaller than window size, the rolling computation fails. This is common with Zarr data chunked by single timestep.

**Impact**: Common time-series workflows with Zarr data fail without explicit rechunking.

**Workaround**: Rechunk before rolling: `ds.chunk({'time': 30}).rolling(time=30).mean()`

**Note**: This may be a dask-level issue with how rolling is lowered in expressions, not xarray-specific.

#### 4. `to_netcdf` with `compute=False` - Tokenization Error

```python
ds.to_netcdf("file.nc", compute=False)
# TokenizationError: Object <xarray.backends.netCDF4_.NetCDF4ArrayWrapper object at ...>
# cannot be deterministically hashed.
```

**Cause**: NetCDF4 backend wrapper objects can't be tokenized by the expression system's deterministic hashing.

**Impact**: Lazy writes to NetCDF fail. (`to_zarr` with `compute=False` works fine.)

**Location**: Likely in `xarray/backends/netCDF4_.py`

### Operations That Work Well

The following operations were tested and work correctly:

- Basic compute of single Dataset/DataArray
- `map_blocks` (with expression-based path)
- `groupby`, `resample` (with `use_flox=True`)
- `rolling` (when chunk size >= window size)
- `where`, `diff`, `coarsen`, `interp`
- `weighted`, `dot`, `polyval`, `quantile`, `median`
- `stack/unstack`, `transpose`, `expand_dims`, `squeeze`
- `sel`, `isel`, `reindex`
- `merge`, `combine_by_coords`, `combine_first`
- `fillna`, `dropna`, `clip`
- `ffill`, `bfill`, `cumsum`, `integrate`
- `differentiate`, `broadcast_like`
- Deep expression trees (50+ chained operations)
- Large number of variables (100+ vars in single Dataset)
- `to_zarr` with `compute=False`
- Fusion and optimization of single-object graphs
- Shared subexpression deduplication within single Dataset

### Operations Not Yet Tested

- `polyfit` (fitting, not just evaluation)
- `curvefit`
- `cov`, `corr` (correlation/covariance)
- Complex multi-dimensional indexing
- `broadcast` with explicit dims
- `pad`
- `shift`, `roll` (dimension shifting)

---

## References

### Dask Repository Files (in `../dask/`)

- `dask/_expr.py` - Base `Expr` class, `_ExprSequence`, optimization methods
- `dask/base.py` - `compute()`, `collections_to_expr()`, `unpack_collections()`
- `dask/array/_array_expr/_expr.py` - `ArrayExpr`, `FinalizeComputeArray`
- `dask/array/_array_expr/_collection.py` - Array collection wrapper
- `dask/_task_spec.py` - `Task`, `TaskRef`, `DataNode` for graph building
- `designs/array-expr.md` - Design principles for expression system

### xarray Repository Files

- `xarray/core/dataset.py` - Dataset class, current `__dask_graph__` at line 634
- `xarray/core/dataarray.py` - DataArray class, `__dask_graph__` at line 1105
- `xarray/namedarray/core.py` - NamedArray with dask protocol methods (lines 589-658)
- `xarray/core/variable.py` - Variable class
- `xarray/namedarray/parallelcompat.py` - ChunkManager abstraction
- `xarray/namedarray/daskmanager.py` - Dask-specific ChunkManager

### Key Patterns to Understand

1. **Expression construction**: Use `_parameters` list, no custom `__init__`
2. **Tokenization**: Deterministic naming for deduplication
3. **Layer generation**: `_layer()` returns dict of tasks
4. **Finalization**: Two-phase pattern (Expr → ExprFinalize)
5. **Task specification**: Use `Task`, `TaskRef`, `DataNode` from `dask/_task_spec.py`
