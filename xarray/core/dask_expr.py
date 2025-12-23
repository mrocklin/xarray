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
    """
    try:
        # Check that dask arrays actually have .expr attribute
        # (the expression system may be present but not active)
        import dask.array as da
        from dask._expr import Expr  # noqa: F401

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

    def _simplify_expr(expr, lower: bool = False):
        """Simplify an expression, optionally lowering to executable form.

        Parameters
        ----------
        expr : Expr
            The expression to simplify
        lower : bool
            If True, also lower the expression (needed for Finalize expressions
            where array exprs need Rechunk -> executable tasks)
        """
        if not isinstance(expr, Expr):
            return expr, False
        result = expr.simplify()
        if lower and hasattr(result, "lower_completely"):
            result = result.lower_completely()
        changed = result._name != expr._name
        return result, changed

    def _collect_expr_dependencies(*expr_sources):
        """Collect Expr objects from multiple sources (single exprs or tuples)."""
        deps = []
        for source in expr_sources:
            if isinstance(source, Expr):
                deps.append(source)
            elif isinstance(source, tuple):
                deps.extend(expr for expr in source if isinstance(expr, Expr))
        return deps

    # --- Visualization helpers ---

    def _format_dims(dims: tuple, shape: tuple) -> str:
        """Format dimensions with sizes, e.g. 'time: 100, lat: 50'."""
        if not dims:
            return "()"
        return ", ".join(f"{d}: {s}" for d, s in zip(dims, shape, strict=False))

    def _format_chunks_summary(chunks: tuple) -> str:
        """Summarize chunks compactly, e.g. '100×50'."""
        if not chunks:
            return ""
        sizes = [c[0] if c else 0 for c in chunks]
        return "×".join(str(s) for s in sizes)

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
        return len(s) >= 8 and all(c in "0123456789abcdef" for c in s.lower())

    def _get_op_name(node) -> str:
        """Get a clean operation name from the expression.

        Extracts the operation type from names like:
        - 'open_dataset-air-311f2fb2...' -> 'Open Dataset'
        - 'random_sample-abc123...' -> 'Random Sample'
        - 'mean-aggregate-def456...' -> 'Mean'
        """
        if hasattr(node, "_name") and "-" in node._name:
            parts = node._name.split("-")
            # Take parts until we hit a hash
            meaningful = []
            for part in parts:
                if _is_hex_hash(part):
                    break
                meaningful.append(part)

            if meaningful:
                # Take just the first part (operation type)
                # Skip suffixes like 'aggregate', 'partial', 'chunk'
                name = meaningful[0]
                for skip in ["aggregate", "partial", "chunk", "agg"]:
                    if name == skip and len(meaningful) > 1:
                        name = meaningful[1]
                        break
                return name.replace("_", " ").title()

        # Fall back to class name
        return type(node).__name__

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
            padding=(0, 2),
            collapse_padding=True,
        )

        table.add_column("Operation", no_wrap=True)
        table.add_column("Dims", no_wrap=True)
        table.add_column("Bytes", justify="right", no_wrap=True)
        table.add_column("Chunks", justify="right", no_wrap=True)

        # Walk the expression tree using shared utility
        nodes = list(
            walk_expr_with_prefix(expr, is_expr_child=lambda op: hasattr(op, "chunks"))
        )

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

            # Format dimensions
            if hasattr(node, "shape"):
                node_dims = dims[: len(node.shape)] if dims else ()
                dim_str = _format_dims(node_dims, node.shape)
            else:
                dim_str = ""

            chunks = node.chunks if hasattr(node, "chunks") else ()

            # Dim data columns for small arrays (operation column stays bright)
            data_style = None if emphasize else "dim"

            table.add_row(
                op_text,
                Text(dim_str, style=data_style),
                Text(format_bytes(nbytes), style=data_style),
                Text(_format_chunks_summary(chunks), style=data_style),
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

        def _simplify_down(self):
            """Simplify nested expressions.

            The optimizer only recurses into direct Expr operands, but we store
            expressions in tuples. This method explicitly simplifies them.
            """
            new_var_exprs = []
            any_changed = False
            for expr in self.var_exprs:
                new_expr, changed = _simplify_expr(expr)
                any_changed = any_changed or changed
                new_var_exprs.append(new_expr)

            new_coord_exprs = []
            for expr in self.coord_exprs:
                new_expr, changed = _simplify_expr(expr)
                any_changed = any_changed or changed
                new_coord_exprs.append(new_expr)

            if any_changed:
                return DatasetExpr(
                    var_names=self.var_names,
                    var_exprs=tuple(new_var_exprs),
                    coord_names=self.coord_names,
                    coord_exprs=tuple(new_coord_exprs),
                    non_chunked_vars=self.non_chunked_vars,
                    non_chunked_coords=self.non_chunked_coords,
                    dims=self.dims,
                    var_dims=self.var_dims,
                    attrs=self.attrs,
                )
            return None

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

        def _simplify_down(self):
            """Simplify and lower nested expressions."""
            new_var_exprs = []
            any_changed = False
            for expr in self.var_exprs:
                new_expr, changed = _simplify_expr(expr, lower=True)
                any_changed = any_changed or changed
                new_var_exprs.append(new_expr)

            new_coord_exprs = []
            for expr in self.coord_exprs:
                new_expr, changed = _simplify_expr(expr, lower=True)
                any_changed = any_changed or changed
                new_coord_exprs.append(new_expr)

            if any_changed:
                return DatasetExprFinalize(
                    var_names=self.var_names,
                    var_exprs=tuple(new_var_exprs),
                    coord_names=self.coord_names,
                    coord_exprs=tuple(new_coord_exprs),
                    non_chunked_vars=self.non_chunked_vars,
                    non_chunked_coords=self.non_chunked_coords,
                    dims=self.dims,
                    var_dims=self.var_dims,
                    attrs=self.attrs,
                )
            return None

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

        def _simplify_down(self):
            """Simplify nested expressions."""
            new_data_expr, any_changed = _simplify_expr(self.data_expr)

            new_coord_exprs = []
            for expr in self.coord_exprs:
                new_expr, changed = _simplify_expr(expr)
                any_changed = any_changed or changed
                new_coord_exprs.append(new_expr)

            if any_changed:
                return DataArrayExpr(
                    name=self.name,
                    data_expr=new_data_expr,
                    coord_names=self.coord_names,
                    coord_exprs=tuple(new_coord_exprs),
                    non_chunked_coords=self.non_chunked_coords,
                    dims=self.dims,
                    attrs=self.attrs,
                )
            return None

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

        def _simplify_down(self):
            """Simplify and lower nested expressions."""
            new_data_expr, any_changed = _simplify_expr(self.data_expr, lower=True)

            new_coord_exprs = []
            for expr in self.coord_exprs:
                new_expr, changed = _simplify_expr(expr, lower=True)
                any_changed = any_changed or changed
                new_coord_exprs.append(new_expr)

            if any_changed:
                return DataArrayExprFinalize(
                    name=self.name,
                    data_expr=new_data_expr,
                    coord_names=self.coord_names,
                    coord_exprs=tuple(new_coord_exprs),
                    non_chunked_coords=self.non_chunked_coords,
                    dims=self.dims,
                    attrs=self.attrs,
                )
            return None

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

else:
    # Fallback for older dask versions without expression support
    DatasetExpr = None  # type: ignore[misc, assignment]
    DataArrayExpr = None  # type: ignore[misc, assignment]
    DatasetExprFinalize = None  # type: ignore[misc, assignment]
    DataArrayExprFinalize = None  # type: ignore[misc, assignment]
