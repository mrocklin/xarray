"""
Dask expression integration for xarray.

This module provides expression classes that allow xarray Dataset and DataArray
to participate in Dask's expression-based optimization pipeline.

Requires dask with array expression support.
"""

from __future__ import annotations

import functools
from typing import Any


def _has_expr_support() -> bool:
    """Check if dask has functional array expression support.

    This checks not just for the presence of expression classes, but that
    dask arrays actually expose the .expr property (which indicates the
    expression system is active, not just present in the codebase).
    Also requires the _rich_table module for visualization.
    """
    try:
        # Check that dask arrays actually have .expr attribute
        # (the expression system may be present but not active)
        import dask.array as da
        from dask._expr import Expr  # noqa: F401
        from dask._rich_table import walk_expr_with_prefix  # noqa: F401

        test_arr = da.ones((2, 2), chunks=1)
        return hasattr(test_arr, "expr")
    except (ImportError, Exception):
        return False


HAS_EXPR_SUPPORT = _has_expr_support()


if HAS_EXPR_SUPPORT:
    from dask._expr import Expr
    from dask._rich_table import (
        REDUCER_COLOR,
        SOURCE_COLOR,
        ExprTable,
        compute_row_emphasis,
        format_bytes,
        get_op_style,
        walk_expr_with_prefix,
    )
    from dask._task_spec import DataNode, Task, TaskRef
    from dask._task_spec import List as TaskList
    from dask.array._array_expr._expr import ArrayExpr

    def _collect_expr_dependencies(*expr_sources):
        """Collect Expr objects from multiple sources (single exprs or tuples)."""
        deps = []
        for source in expr_sources:
            if isinstance(source, Expr):
                deps.append(source)
            elif isinstance(source, tuple):
                deps.extend(expr for expr in source if isinstance(expr, Expr))
        return deps

    def _fuse_exprs(exprs: list) -> list:
        """Fuse expressions jointly via _ExprSequence to preserve sharing.

        Parameters
        ----------
        exprs : list
            List of expressions (some may not have .fuse())

        Returns
        -------
        list
            Fused expressions in same order, non-fusable unchanged
        """
        from dask._expr import _ExprSequence

        fusable = [e for e in exprs if hasattr(e, "fuse")]
        if not fusable:
            return exprs

        # Fuse jointly
        if len(fusable) == 1:
            fused = [fusable[0].fuse()]
        else:
            seq = _ExprSequence(*fusable)
            fused_seq = fusable[0].fuse.__func__(seq)
            if isinstance(fused_seq, _ExprSequence):
                fused = list(fused_seq.operands)
            else:
                fused = [e.fuse() for e in fusable]

        # Map back to original order
        fused_iter = iter(fused)
        return [next(fused_iter) if hasattr(e, "fuse") else e for e in exprs]

    # --- Visualization helpers ---

    def _format_shape(shape: tuple) -> str:
        """Format shape compactly, e.g. '(4384, 121, 281)'."""
        if not shape:
            return "()"
        return "×".join(str(s) for s in shape)

    def _get_expr_nbytes(expr) -> float:
        """Get nbytes for an array expression."""
        import math

        try:
            shape = expr.shape if hasattr(expr, "shape") else ()
            dtype = expr.dtype if hasattr(expr, "dtype") else None
            if not shape or dtype is None:
                return math.nan
            if any(math.isnan(s) for s in shape):
                return math.nan
            return math.prod(shape) * dtype.itemsize
        except Exception:
            return math.nan

    class _MultiTableWrapper:
        """Wrapper for multiple rich Tables (for Dataset with multiple variables)."""

        def __init__(self, tables: list, title: str = ""):
            self._tables = tables
            self._title = title
            self._html_cache = None
            self._text_cache = None

        def _print_content(self, console):
            if self._title:
                console.print(f"[bold]{self._title}[/bold]")
                console.print()
            for table in self._tables:
                console.print(table)
                console.print()

        def _repr_html_(self):
            if self._html_cache is None:
                import io

                from rich.console import Console

                console = Console(
                    file=io.StringIO(),
                    force_terminal=False,
                    force_jupyter=False,
                    record=True,
                )
                self._print_content(console)
                self._html_cache = console.export_html(
                    inline_styles=True, code_format="<pre>{code}</pre>"
                )
            return self._html_cache

        def __repr__(self):
            if self._text_cache is None:
                import io

                from rich.console import Console

                console = Console(
                    file=io.StringIO(), force_terminal=True, force_jupyter=False
                )
                self._print_content(console)
                self._text_cache = console.file.getvalue().rstrip()
            return self._text_cache

        def _repr_mimebundle_(self, **kwargs):
            return {"text/html": self._repr_html_()}

        def print(self):
            from rich.console import Console

            self._print_content(Console())

    def _get_op_color(node) -> str | None:
        """Determine operation color based on node type."""
        # Check if it's a source (no array dependencies)
        if hasattr(node, "operands"):
            deps = [op for op in node.operands if hasattr(op, "chunks")]
            if not deps:
                return SOURCE_COLOR

        # Check for reducer patterns in the name
        if hasattr(node, "_name"):
            name_lower = node._name.lower()
            if any(
                r in name_lower
                for r in ["reduce", "sum", "mean", "max", "min", "std", "var", "agg"]
            ):
                return REDUCER_COLOR

        return None

    def _is_hex_hash(s: str) -> bool:
        """Check if string looks like a hex hash (8+ hex chars)."""
        return len(s) >= 8 and all(c in "0123456789abcdefABCDEF" for c in s)

    def _get_op_name(node) -> str:
        """Get a clean operation name from the expression.

        Uses dask's key_split to strip hashes, then extracts the operation type.
        Examples:
        - '_trim-9fd11138...' -> 'Trim'
        - 'Trim 9Fd11138...' -> 'Trim'
        - 'open_dataset-air-311f2fb2' -> 'Open Dataset'
        - 'mean-aggregate-def456' -> 'Mean'
        - FusedBlockwise -> 'Fused'
        """
        from dask.base import key_split

        # Check for FusedBlockwise type
        if type(node).__name__ == "FusedBlockwise":
            return "Fused"

        if not hasattr(node, "_name"):
            return type(node).__name__

        name = node._name

        # Handle space-separated hashes first (e.g., "Trim 9Fd11138...")
        if " " in name:
            parts = [p for p in name.split(" ") if p and not _is_hex_hash(p)]
            name = "-".join(parts) if parts else name

        # Use dask's key_split to strip trailing hashes
        name = key_split(name)

        # Take first part to avoid dask internals like -aggregate, -partial
        if "-" in name:
            name = name.split("-")[0]

        return name.replace("_", " ").title()

    def _build_array_table(expr, dims: tuple, title: str | None = None):
        """Build a rich Table for a single array expression."""
        from rich.table import Table
        from rich.text import Text

        table = Table(
            title=title,
            title_justify="left",
            show_header=True,
            header_style="dim",
            box=None,
            padding=(0, 1),
            collapse_padding=True,
        )

        table.add_column("Operation", no_wrap=True)
        table.add_column("Shape", no_wrap=True)
        table.add_column("Bytes", justify="right", no_wrap=True)

        # Walk the expression tree using shared utility
        # Filter to nodes that have both operands (for walking) and chunks (for display)
        def is_array_expr(op):
            return hasattr(op, "chunks") and hasattr(op, "operands")

        nodes = list(walk_expr_with_prefix(expr, is_expr_child=is_array_expr))

        # Compute row emphasis based on relative bytes (dim small rows)
        node_bytes = [_get_expr_nbytes(n) for n, _ in nodes]
        row_emphasis = compute_row_emphasis(node_bytes)

        for (node, prefix), nbytes, emphasize in zip(
            nodes, node_bytes, row_emphasis, strict=True
        ):
            op_name = _get_op_name(node)
            color = _get_op_color(node)

            op_text = Text()
            op_text.append(prefix, style="dim")
            op_text.append(op_name, style=get_op_style(color))

            # Format shape
            shape = node.shape if hasattr(node, "shape") else ()
            shape_str = _format_shape(shape)

            # Dim data columns for small arrays (operation column stays bright)
            data_style = None if emphasize else "dim"

            table.add_row(
                op_text,
                Text(shape_str, style=data_style),
                Text(format_bytes(nbytes), style=data_style),
            )

        return table

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
            Names of chunked coordinates (in order)
        coord_exprs : tuple[Expr, ...]
            Expressions for each chunked coordinate
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
        ]
        _defaults = {
            "non_chunked_vars": {},
            "non_chunked_coords": {},
            "attrs": None,
        }

        @functools.cached_property
        def _name(self) -> str:
            return f"dataset-{self.deterministic_token}"

        def dependencies(self) -> list[Expr]:
            """Return all variable and coordinate expressions."""
            return _collect_expr_dependencies(self.var_exprs, self.coord_exprs)

        def _layer(self) -> dict:
            """Container node - no tasks of its own.

            The base Expr.__dask_graph__() traverses dependencies() and
            collects _layer() from each child, so we don't duplicate that here.
            """
            return {}

        # No _simplify_down or _lower needed - dask walks tuples when dependencies() is overridden

        def __dask_keys__(self) -> list:
            """Return keys for all chunked variables and coordinates.

            Structure: [[var1_keys...], [var2_keys...], [coord1_keys...], ...]
            """
            keys = [list(expr.__dask_keys__()) for expr in self.var_exprs]
            keys.extend(list(expr.__dask_keys__()) for expr in self.coord_exprs)
            return keys

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
            )

        def fuse(self):
            """Fuse nested array expressions jointly to preserve sharing."""
            n_vars = len(self.var_exprs)
            all_exprs = list(self.var_exprs) + list(self.coord_exprs)
            fused = _fuse_exprs(all_exprs)

            return DatasetExpr(
                var_names=self.var_names,
                var_exprs=tuple(fused[:n_vars]),
                coord_names=self.coord_names,
                coord_exprs=tuple(fused[n_vars:]),
                non_chunked_vars=self.non_chunked_vars,
                non_chunked_coords=self.non_chunked_coords,
                dims=self.dims,
                var_dims=self.var_dims,
                attrs=self.attrs,
            )

        def _table(self):
            """Build rich tables for all variables."""
            tables = []
            for name, expr in zip(self.var_names, self.var_exprs, strict=True):
                dims = self.var_dims.get(name, ())
                tables.append(_build_array_table(expr, dims, title=name))
            for name, expr in zip(self.coord_names, self.coord_exprs, strict=True):
                dims = self.var_dims.get(name, (name,))
                tables.append(_build_array_table(expr, dims, title=f"{name} (coord)"))
            return _MultiTableWrapper(tables, title="DatasetExpr")

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
        ]
        _defaults = {
            "non_chunked_vars": {},
            "non_chunked_coords": {},
            "attrs": None,
        }

        @functools.cached_property
        def _name(self) -> str:
            return f"dataset-finalize-{self.deterministic_token}"

        def dependencies(self) -> list[Expr]:
            return _collect_expr_dependencies(self.var_exprs, self.coord_exprs)

        # No _simplify_down or _lower needed - dask walks tuples when dependencies() is overridden

        def fuse(self):
            """Fuse nested array expressions jointly to preserve sharing."""
            n_vars = len(self.var_exprs)
            all_exprs = list(self.var_exprs) + list(self.coord_exprs)
            fused = _fuse_exprs(all_exprs)

            return DatasetExprFinalize(
                var_names=self.var_names,
                var_exprs=tuple(fused[:n_vars]),
                coord_names=self.coord_names,
                coord_exprs=tuple(fused[n_vars:]),
                non_chunked_vars=self.non_chunked_vars,
                non_chunked_coords=self.non_chunked_coords,
                dims=self.dims,
                var_dims=self.var_dims,
                attrs=self.attrs,
            )

        def _layer(self) -> dict:
            """Build layer with reconstruction task."""
            from dask.base import flatten

            # Collect keys from finalized expressions
            # After finalize_compute and simplification, arrays have single chunk
            # but keys may still be nested [[key]] for multi-dimensional arrays
            var_keys = []
            for expr in self.var_exprs:
                keys = list(flatten(expr.__dask_keys__()))
                # Should be single key after finalize
                var_keys.append(keys[0] if keys else None)

            coord_keys = []
            for expr in self.coord_exprs:
                keys = list(flatten(expr.__dask_keys__()))
                coord_keys.append(keys[0] if keys else None)

            all_keys = var_keys + coord_keys

            # Build reconstruction task
            return {
                self._name: Task(
                    self._name,
                    _reconstruct_dataset,
                    TaskList(*[TaskRef(k) for k in all_keys]),
                    DataNode(None, self.var_names),
                    DataNode(None, self.coord_names),
                    DataNode(None, self.non_chunked_vars),
                    DataNode(None, self.non_chunked_coords),
                    DataNode(None, self.var_dims),
                    DataNode(None, self.attrs),
                )
            }

        def __dask_keys__(self) -> list:
            """Return single key for the reconstructed Dataset."""
            return [self._name]

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
        data_vars = {
            name: (var_dims.get(name, ()), arr)
            for name, arr in zip(var_names, var_arrays, strict=True)
        }
        # Add non-chunked variables
        data_vars.update(non_chunked_vars)

        # Build coords dict
        coords = {
            name: (var_dims.get(name, (name,)), arr)
            for name, arr in zip(coord_names, coord_arrays, strict=True)
        }
        # Add non-chunked coordinates
        coords.update(non_chunked_coords)

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
        _defaults = {
            "name": None,
            "non_chunked_coords": {},
            "attrs": None,
        }

        @functools.cached_property
        def _name(self) -> str:
            return f"dataarray-{self.deterministic_token}"

        def dependencies(self) -> list[Expr]:
            return _collect_expr_dependencies(self.data_expr, self.coord_exprs)

        def _layer(self) -> dict:
            """Container node - no tasks of its own."""
            return {}

        # No _simplify_down or _lower needed - dask walks tuples when dependencies() is overridden

        def __dask_keys__(self) -> list:
            keys = [list(self.data_expr.__dask_keys__())]
            keys.extend(list(expr.__dask_keys__()) for expr in self.coord_exprs)
            return keys

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

        def fuse(self):
            """Fuse nested array expressions jointly to preserve sharing."""
            all_exprs = [self.data_expr] + list(self.coord_exprs)
            fused = _fuse_exprs(all_exprs)

            return DataArrayExpr(
                name=self.name,
                data_expr=fused[0],
                coord_names=self.coord_names,
                coord_exprs=tuple(fused[1:]),
                non_chunked_coords=self.non_chunked_coords,
                dims=self.dims,
                attrs=self.attrs,
            )

        def _table(self):
            """Build rich table for the data array."""
            return ExprTable(_build_array_table(self.data_expr, self.dims))

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
        _defaults = {
            "name": None,
            "non_chunked_coords": {},
            "attrs": None,
        }

        @functools.cached_property
        def _name(self) -> str:
            return f"dataarray-finalize-{self.deterministic_token}"

        def dependencies(self) -> list[Expr]:
            return _collect_expr_dependencies(self.data_expr, self.coord_exprs)

        # No _simplify_down or _lower needed - dask walks tuples when dependencies() is overridden

        def fuse(self):
            """Fuse nested array expressions jointly to preserve sharing."""
            all_exprs = [self.data_expr] + list(self.coord_exprs)
            fused = _fuse_exprs(all_exprs)

            return DataArrayExprFinalize(
                name=self.name,
                data_expr=fused[0],
                coord_names=self.coord_names,
                coord_exprs=tuple(fused[1:]),
                non_chunked_coords=self.non_chunked_coords,
                dims=self.dims,
                attrs=self.attrs,
            )

        def _layer(self) -> dict:
            from dask.base import flatten

            # Get key for data (single key after finalize)
            data_keys = list(flatten(self.data_expr.__dask_keys__()))
            data_key = data_keys[0] if data_keys else None

            coord_keys = []
            for expr in self.coord_exprs:
                keys = list(flatten(expr.__dask_keys__()))
                if keys:
                    coord_keys.append(keys[0])

            return {
                self._name: Task(
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
            }

        def __dask_keys__(self) -> list:
            return [self._name]

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
        coords = dict(zip(coord_names, coord_arrays, strict=True))
        # Add non-chunked coordinates
        coords.update(
            {
                cname: ((cdims, cdata) if cdims else cdata)
                for cname, (cdims, cdata) in non_chunked_coords.items()
            }
        )

        return xr.DataArray(data, dims=dims, coords=coords, name=name, attrs=attrs)

    # =========================================================================
    # map_blocks expression classes
    # =========================================================================

    class MapBlocksSharedExpr(Expr):
        """Expression representing the shared computation of a map_blocks operation.

        This expression generates the tasks that call the user function on
        each chunk of the input. Each task produces a dict/Dataset result.
        Individual output variables are extracted by MapBlocksVarExpr.

        Parameters
        ----------
        func : callable
            The user function to apply
        gname : str
            Base name for output tasks (e.g., "my_func-abc123")
        input_var_exprs : tuple[Expr, ...]
            Expressions for each input variable (flat tuple for optimizer)
        input_var_meta : tuple
            Tuple of (var_name, dims, attrs) for each input variable
        input_coord_exprs : tuple[Expr, ...]
            Expressions for each input coordinate (flat tuple for optimizer)
        input_coord_meta : tuple
            Tuple of (coord_name, dims, attrs) for each input coordinate
        non_chunked_info : tuple
            Tuple of (name, dims, data, attrs, is_coord) for non-chunked variables
        input_chunks : tuple
            Chunk info from inputs as tuple of (dim, chunks) pairs
        input_chunk_bounds : tuple
            Chunk bounds as tuple of (dim, bounds) pairs
        is_array_flags : tuple
            Which args are DataArrays (for conversion)
        expected_shapes : tuple
            Expected output shapes as tuple of (dim, size) pairs
        expected_data_vars : tuple
            Expected data variable names
        expected_coords : tuple
            Expected coordinate names
        indexes_info : tuple
            Tuple of (name, index_token) for indexes
        kwargs : dict
            Keyword arguments to pass to func
        dataset_attrs : dict | None
            Attributes from input dataset
        """

        # Enable optimization traversal into input_var_exprs and input_coord_exprs tuples
        _optimize_list_operands = True

        _parameters = [
            "func",
            "gname",
            "input_var_exprs",
            "input_var_meta",
            "input_coord_exprs",
            "input_coord_meta",
            "non_chunked_info",
            "input_chunks",
            "input_chunk_bounds",
            "is_array_flags",
            "expected_shapes",
            "expected_data_vars",
            "expected_coords",
            "indexes_info",
            "kwargs",
            "dataset_attrs",
        ]
        _defaults = {
            "indexes_info": (),
            "kwargs": None,
            "dataset_attrs": None,
        }

        @functools.cached_property
        def _name(self) -> str:
            return f"{self.gname}-{self.deterministic_token}"

        def dependencies(self) -> list[Expr]:
            """Return input variable and coordinate expressions."""
            deps = list(self.input_var_exprs) + list(self.input_coord_exprs)
            return [d for d in deps if isinstance(d, Expr)]

        def _layer(self) -> dict:
            """Generate the shared map_blocks tasks.

            These tasks call _wrapper and produce dicts of results.
            """
            from xarray.core.parallel import _wrapper

            graph: dict = {}
            input_chunks = dict(self.input_chunks)
            input_chunk_bounds = dict(self.input_chunk_bounds)

            # Build expected dict for _wrapper
            expected_base: dict = {
                "data_vars": set(self.expected_data_vars),
                "coords": set(self.expected_coords),
            }

            # Coord names set
            coord_names = {name for name, _, _ in self.input_coord_meta}
            for name, _dims, _data, _attrs, is_coord in self.non_chunked_info:
                if is_coord:
                    coord_names.add(name)

            # Iterate over all chunk combinations
            ichunk = {dim: range(len(chunks)) for dim, chunks in input_chunks.items()}
            import itertools

            for chunk_tuple in itertools.product(*ichunk.values()):
                chunk_index = dict(zip(ichunk.keys(), chunk_tuple, strict=True))

                # Build blocked_args (the subset task references)
                data_vars = []
                coords = []

                # Process chunked variables
                for expr, (var_name, dims, attrs) in zip(
                    self.input_var_exprs, self.input_var_meta, strict=True
                ):
                    expr_name = expr._name
                    chunk_key = (expr_name,) + tuple(chunk_index[dim] for dim in dims)
                    chunk_var_task = (
                        f"{var_name}-{self.gname}-{expr_name!r}",
                    ) + chunk_tuple
                    graph[chunk_var_task] = (tuple, [dims, chunk_key, attrs])
                    data_vars.append([var_name, chunk_var_task])

                # Process chunked coordinates
                for expr, (coord_name, dims, attrs) in zip(
                    self.input_coord_exprs, self.input_coord_meta, strict=True
                ):
                    expr_name = expr._name
                    chunk_key = (expr_name,) + tuple(chunk_index[dim] for dim in dims)
                    chunk_var_task = (
                        f"{coord_name}-{self.gname}-{expr_name!r}",
                    ) + chunk_tuple
                    graph[chunk_var_task] = (tuple, [dims, chunk_key, attrs])
                    coords.append([coord_name, chunk_var_task])

                # Process non-chunked variables
                from xarray.core.parallel import _get_chunk_slicer

                for name, dims, data, attrs, is_coord in self.non_chunked_info:
                    # Subset the data for this chunk
                    subsetter = {
                        dim: _get_chunk_slicer(dim, chunk_index, input_chunk_bounds)
                        for dim in dims
                    }
                    from dask.base import tokenize

                    chunk_dims_set = set(chunk_index)
                    if set(dims) < chunk_dims_set:
                        this_var_chunk_tuple = tuple(chunk_index[dim] for dim in dims)
                    else:
                        this_var_chunk_tuple = chunk_tuple

                    chunk_var_task = (
                        f"{name}-{self.gname}-{tokenize(subsetter)}",
                    ) + this_var_chunk_tuple

                    # Only add if not already present (dimension coords)
                    if len(dims) == 0 or chunk_var_task not in graph:
                        if len(dims) == 0:
                            subset_data = data
                        else:
                            import numpy as np

                            slices = tuple(subsetter.get(d, slice(None)) for d in dims)
                            if isinstance(data, np.ndarray):
                                subset_data = data[slices]
                            else:
                                subset_data = data

                        # For scalars, dims is ()
                        subset_dims = dims
                        graph[chunk_var_task] = (
                            tuple,
                            [subset_dims, subset_data, attrs],
                        )

                    if is_coord:
                        coords.append([name, chunk_var_task])
                    else:
                        data_vars.append([name, chunk_var_task])

                # Build expected for this chunk
                expected = {
                    **expected_base,
                    "shapes": {
                        k: input_chunks[k][v]
                        for k, v in chunk_index.items()
                        if k in self.expected_shapes
                    },
                }

                # Build the index lookup dict for this chunk
                indexes_dict = dict(self.indexes_info)

                # Create the wrapper task
                from xarray.core.dataset import Dataset

                blocked_arg = (
                    Dataset,
                    (dict, data_vars),
                    (dict, coords),
                    self.dataset_attrs,
                )
                from_wrapper = (self.gname,) + chunk_tuple
                graph[from_wrapper] = (
                    _wrapper,
                    self.func,
                    [blocked_arg],
                    self.kwargs if self.kwargs else {},
                    self.is_array_flags,
                    expected,
                    indexes_dict,
                )

            return graph

        def __dask_keys__(self) -> list:
            """Return keys for all wrapper tasks."""
            input_chunks = dict(self.input_chunks)
            ichunk = {dim: range(len(chunks)) for dim, chunks in input_chunks.items()}
            import itertools

            return [
                (self.gname,) + chunk_tuple
                for chunk_tuple in itertools.product(*ichunk.values())
            ]

        # No need for _simplify_down or _lower - dask's optimizer now walks
        # through tuples/lists looking for Exprs when we have custom dependencies()

    class MapBlocksVarExpr(ArrayExpr):
        """Expression for extracting one variable from a map_blocks result.

        This is an ArrayExpr that represents one output variable
        from the map_blocks operation.

        Parameters
        ----------
        shared_expr : MapBlocksSharedExpr
            The shared expression that generates the wrapper tasks
        var_name : str
            Name of the variable to extract
        var_dims : tuple
            Dimension names for this variable
        chunks : tuple
            Chunk sizes for each dimension
        dtype : dtype
            Data type of the output
        meta_array : array-like
            Metadata array (renamed from _meta to avoid conflict)
        """

        _parameters = [
            "shared_expr",
            "var_name",
            "var_dims",
            "chunks",
            "dtype",
            "meta_array",
        ]

        @functools.cached_property
        def _name(self) -> str:
            return f"{self.var_name}-{self.shared_expr._name}"

        @functools.cached_property
        def _meta(self):
            """Return metadata array."""
            return self.meta_array

        @functools.cached_property
        def shape(self):
            from dask.utils import cached_cumsum

            return tuple(cached_cumsum(c, initial_zero=True)[-1] for c in self.chunks)

        @functools.cached_property
        def ndim(self):
            return len(self.shape)

        @functools.cached_property
        def npartitions(self):
            """Total number of partitions (chunks)."""
            from functools import reduce
            from operator import mul

            return reduce(mul, [len(c) for c in self.chunks], 1)

        def dependencies(self) -> list[Expr]:
            """Return the shared expression as dependency."""
            return [self.shared_expr]

        def _layer(self) -> dict:
            """Generate getitem tasks to extract this variable."""
            import operator

            graph = {}
            input_chunks = dict(self.shared_expr.input_chunks)
            ichunk = {dim: range(len(chunks)) for dim, chunks in input_chunks.items()}

            import itertools

            for chunk_tuple in itertools.product(*ichunk.values()):
                chunk_index = dict(zip(ichunk.keys(), chunk_tuple, strict=True))

                # Build the output key for this variable
                out_key = (self._name,) + tuple(
                    chunk_index.get(dim, 0) for dim in self.var_dims
                )

                # Reference the shared task (use gname, which is the stable key)
                from_wrapper = (self.shared_expr.gname,) + chunk_tuple

                graph[out_key] = (operator.getitem, from_wrapper, self.var_name)

            return graph

        def __dask_keys__(self) -> list:
            """Return keys for this variable's chunks."""
            from itertools import product

            return [
                (self._name,) + block_id
                for block_id in product(*[range(len(c)) for c in self.chunks])
            ]

        def finalize_compute(self):
            """For compatibility with the finalize pattern."""
            from dask.array._array_expr._expr import FinalizeComputeArray

            return FinalizeComputeArray(self)

        # No need for _simplify_down or _lower - shared_expr is a direct operand
        # so dask's optimizer traverses it automatically


else:
    # Fallback for older dask versions without expression support
    DatasetExpr = None  # type: ignore[misc, assignment]
    DataArrayExpr = None  # type: ignore[misc, assignment]
    DatasetExprFinalize = None  # type: ignore[misc, assignment]
    DataArrayExprFinalize = None  # type: ignore[misc, assignment]
    MapBlocksSharedExpr = None  # type: ignore[misc, assignment]
    MapBlocksVarExpr = None  # type: ignore[misc, assignment]
