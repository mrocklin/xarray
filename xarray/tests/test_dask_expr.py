"""Tests for dask expression integration."""

import numpy as np
import pytest

dask = pytest.importorskip("dask")
da = pytest.importorskip("dask.array")

import xarray as xr
from xarray.core.dask_expr import HAS_EXPR_SUPPORT

pytestmark = pytest.mark.skipif(
    not HAS_EXPR_SUPPORT, reason="Requires dask with expression support"
)


class TestDatasetExpr:
    def test_expr_property_exists(self):
        """Test that .expr property exists and returns correct type."""
        data = da.ones((10, 10), chunks=5)
        ds = xr.Dataset({"a": (["x", "y"], data)})

        from xarray.core.dask_expr import DatasetExpr

        expr = ds.expr
        assert isinstance(expr, DatasetExpr)

    def test_basic_compute(self):
        """Test computing a Dataset with .expr integration."""
        data = da.ones((10, 10), chunks=5)
        ds = xr.Dataset({"a": (["x", "y"], data)})

        # Dataset.expr is used automatically by dask.compute via collections_to_expr
        (result,) = dask.compute(ds)

        assert isinstance(result, xr.Dataset)
        assert "a" in result
        np.testing.assert_array_equal(result["a"].values, np.ones((10, 10)))

    def test_multiple_variables(self):
        data1 = da.ones((10, 10), chunks=5)
        data2 = da.zeros((10, 10), chunks=5)
        ds = xr.Dataset({"a": (["x", "y"], data1), "b": (["x", "y"], data2)})

        (result,) = dask.compute(ds)

        assert isinstance(result, xr.Dataset)
        assert set(result.data_vars) == {"a", "b"}
        np.testing.assert_array_equal(result["a"].values, np.ones((10, 10)))
        np.testing.assert_array_equal(result["b"].values, np.zeros((10, 10)))

    def test_preserves_attrs(self):
        data = da.ones((10,), chunks=5)
        ds = xr.Dataset({"a": (["x"], data)}, attrs={"description": "test dataset"})

        (result,) = dask.compute(ds)
        assert result.attrs["description"] == "test dataset"

    def test_non_chunked_variables(self):
        chunked = da.ones((10,), chunks=5)
        non_chunked = np.array([1, 2, 3])

        ds = xr.Dataset(
            {
                "chunked": (["x"], chunked),
                "non_chunked": (["y"], non_chunked),
            }
        )

        (result,) = dask.compute(ds)

        np.testing.assert_array_equal(result["chunked"].values, np.ones(10))
        np.testing.assert_array_equal(result["non_chunked"].values, [1, 2, 3])

    def test_coordinates_preserved(self):
        data = da.ones((10, 10), chunks=5)
        x_coord = da.arange(10, chunks=5)
        y_coord = np.arange(10)  # Non-chunked coord

        ds = xr.Dataset({"a": (["x", "y"], data)}, coords={"x": x_coord, "y": y_coord})

        (result,) = dask.compute(ds)

        np.testing.assert_array_equal(result.coords["x"].values, np.arange(10))
        np.testing.assert_array_equal(result.coords["y"].values, np.arange(10))

    def test_error_non_dask_dataset(self):
        ds = xr.Dataset({"a": (["x"], np.ones(10))})

        with pytest.raises(ValueError, match="no chunked variables"):
            _ = ds.expr


class TestDataArrayExpr:
    def test_expr_property_exists(self):
        """Test that .expr property exists and returns correct type."""
        data = da.ones((10, 10), chunks=5)
        arr = xr.DataArray(data, dims=["x", "y"], name="test")

        from xarray.core.dask_expr import DataArrayExpr

        expr = arr.expr
        assert isinstance(expr, DataArrayExpr)

    def test_basic_compute(self):
        data = da.ones((10, 10), chunks=5)
        arr = xr.DataArray(data, dims=["x", "y"], name="test")

        (result,) = dask.compute(arr)

        assert isinstance(result, xr.DataArray)
        assert result.name == "test"
        np.testing.assert_array_equal(result.values, np.ones((10, 10)))

    def test_preserves_dims(self):
        data = da.ones((5, 10, 15), chunks=5)
        arr = xr.DataArray(data, dims=["time", "x", "y"])

        (result,) = dask.compute(arr)

        assert result.dims == ("time", "x", "y")
        assert result.shape == (5, 10, 15)

    def test_preserves_attrs(self):
        data = da.ones((10,), chunks=5)
        arr = xr.DataArray(data, dims=["x"], attrs={"units": "meters"})

        (result,) = dask.compute(arr)
        assert result.attrs["units"] == "meters"

    def test_coordinates_preserved(self):
        data = da.ones((10,), chunks=5)
        arr = xr.DataArray(data, dims=["x"], coords={"x": np.arange(10)}, name="test")

        (result,) = dask.compute(arr)

        np.testing.assert_array_equal(result.coords["x"].values, np.arange(10))

    def test_error_non_dask_dataarray(self):
        arr = xr.DataArray(np.ones(10), dims=["x"])

        with pytest.raises(ValueError, match="not chunked"):
            _ = arr.expr


class TestJointOptimization:
    def test_compute_dataset_with_dask_array(self):
        """Verify Dataset and raw dask arrays can be computed together."""
        base = da.random.random((10, 10), chunks=5)

        ds = xr.Dataset({"a": (["x", "y"], base + 1)})
        arr_mean = base.mean()

        # Both should be optimized together
        result_ds, result_mean = dask.compute(ds, arr_mean)

        assert isinstance(result_ds, xr.Dataset)
        assert isinstance(result_mean, float)

    def test_compute_multiple_datasets(self):
        """Verify multiple Datasets can be computed together."""
        data1 = da.ones((10, 10), chunks=5)
        data2 = da.zeros((10, 10), chunks=5)

        ds1 = xr.Dataset({"a": (["x", "y"], data1)})
        ds2 = xr.Dataset({"b": (["x", "y"], data2)})

        result1, result2 = dask.compute(ds1, ds2)

        assert isinstance(result1, xr.Dataset)
        assert isinstance(result2, xr.Dataset)
        assert "a" in result1
        assert "b" in result2

    def test_joint_compute_multiple_dataarrays(self):
        """Test joint compute of multiple DataArrays."""
        da1 = xr.DataArray(np.ones(10), dims=["x"]).chunk({"x": 5})
        da2 = xr.DataArray(np.ones(10) * 2, dims=["x"]).chunk({"x": 5})

        result1, result2 = dask.compute(da1, da2)

        assert isinstance(result1, xr.DataArray)
        assert isinstance(result2, xr.DataArray)
        np.testing.assert_array_equal(result1.values, np.ones(10))
        np.testing.assert_array_equal(result2.values, np.ones(10) * 2)

    def test_joint_compute_mixed_dataset_dataarray(self):
        """Test joint compute of Dataset and DataArray together."""
        ds = xr.Dataset({"a": (["x"], np.ones(10))}).chunk({"x": 5})
        da = xr.DataArray(np.ones(10) * 3, dims=["x"]).chunk({"x": 5})

        result_ds, result_da = dask.compute(ds, da)

        assert isinstance(result_ds, xr.Dataset)
        assert isinstance(result_da, xr.DataArray)
        np.testing.assert_array_equal(result_ds["a"].values, np.ones(10))
        np.testing.assert_array_equal(result_da.values, np.ones(10) * 3)

    def test_joint_compute_many_objects(self):
        """Test joint compute of many xarray objects."""
        objects = [
            xr.Dataset({"v": (["x"], np.ones(10) * i)}).chunk({"x": 5})
            for i in range(5)
        ]

        results = dask.compute(*objects)

        assert len(results) == 5
        for i, result in enumerate(results):
            assert isinstance(result, xr.Dataset)
            np.testing.assert_array_equal(result["v"].values, np.ones(10) * i)


class TestSharedSubexpressions:
    def test_shared_computation_not_duplicated(self):
        """Verify that shared computation between variables is computed once."""
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

        # Compute - should compute base_computed only once
        (result,) = dask.compute(ds)

        # base_computed has 4 chunks (2x2), so expensive_op should run 4 times
        # NOT 8 times (which would happen if 'a' and 'b' each triggered it)
        assert call_count[0] == 4, (
            f"Expected 4 calls, got {call_count[0]} - shared expr was duplicated!"
        )

        # Verify results are correct
        expected = np.ones((10, 10)) * 2  # base after expensive_op
        np.testing.assert_array_equal(result["a"].values, expected + 1)
        np.testing.assert_array_equal(result["b"].values, expected + 2)

    def test_joint_optimization_with_dask_array(self):
        """Verify computation works when mixing Dataset and raw dask arrays.

        Note: Due to _ExprSequence.fuse() grouping by module, xarray expressions
        and raw dask arrays are fused separately. This means shared subexpressions
        between the two types are not currently deduplicated. This test verifies
        correct results despite the separate fusion.
        """
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

        # Compute together - should produce correct results
        result_ds, result_arr = dask.compute(ds, arr)

        # Verify correct results
        np.testing.assert_array_equal(result_ds["a"].values, np.ones((10, 10)) * 2)
        np.testing.assert_equal(result_arr, 1.0)

        # Note: Due to separate fusion groups, base is computed twice (8 calls)
        # rather than once (4 calls). This is a known limitation.
        assert call_count[0] == 8, f"Expected 8 (known limitation), got {call_count[0]}"


class TestExpressionFusion:
    """Tests for blockwise fusion in xarray expressions."""

    def test_dataset_fusion_reduces_depth_and_tasks(self):
        """Verify that fusion reduces expression depth and task count."""
        # Create dataset and apply chain of fusable xarray operations
        data = da.ones((10, 10), chunks=5)
        ds = xr.Dataset({"a": (["x", "y"], data)})
        ds = ds + 1 + 2 + 3 + 4 + 5  # 5 additions on the Dataset

        expr = ds.expr
        fused = expr.fuse()

        # Fusion should reduce depth
        assert fused._depth() < expr._depth()

        # Fusion should reduce task count
        tasks_before = len(expr.__dask_graph__())
        tasks_after = len(fused.__dask_graph__())
        assert tasks_after < tasks_before, (
            f"Fusion did not reduce tasks: {tasks_before} -> {tasks_after}"
        )

        # Verify the result is still correct
        expected = xr.Dataset({"a": (["x", "y"], np.ones((10, 10)) + 15)})
        xr.testing.assert_equal(ds, expected)

    def test_dataarray_fusion_reduces_depth_and_tasks(self):
        """Verify that fusion reduces expression depth and task count for DataArray."""
        data = da.ones((10, 10), chunks=5)
        arr = xr.DataArray(data, dims=["x", "y"], name="test")
        arr = arr * 2 * 3 * 4 * 5  # 4 multiplications on the DataArray

        expr = arr.expr
        fused = expr.fuse()

        assert fused._depth() < expr._depth()
        assert len(fused.__dask_graph__()) < len(expr.__dask_graph__())

        # Verify correctness
        expected = xr.DataArray(np.ones((10, 10)) * 120, dims=["x", "y"], name="test")
        xr.testing.assert_equal(arr, expected)

    def test_optimize_with_fuse_flag(self):
        """Verify optimize(fuse=True) triggers fusion."""
        data = da.ones((10, 10), chunks=5)
        ds = xr.Dataset({"a": (["x", "y"], data)})
        ds = ds + 1 + 2 + 3  # Operations on the Dataset

        expr = ds.expr
        optimized = expr.optimize(fuse=True)

        # After optimize(fuse=True), inner expressions should be FusedBlockwise
        inner_type = type(optimized.var_exprs[0]).__name__
        assert inner_type == "FusedBlockwise", (
            f"Expected FusedBlockwise, got {inner_type}"
        )

    def test_joint_fusion_preserves_shared_base_in_graph(self):
        """Verify joint fusion keeps shared base as separate tasks in graph.

        With joint fusion via _ExprSequence, a shared base computation should
        appear as separate tasks in the graph (not embedded into each variable's
        fused block). This is important for expensive operations like I/O.
        """
        data = da.ones((4, 4), chunks=2)
        base = data + 1  # shared base
        ds = xr.Dataset(
            {
                "a": (["x", "y"], base + 10),
                "b": (["x", "y"], base + 20),
            }
        )

        fused = ds.expr.fuse()
        graph = fused.__dask_graph__()

        # Get unique key names (without chunk indices)
        key_names = set(k[0] if isinstance(k, tuple) else k for k in graph.keys())

        # Joint fusion should produce 3 unique key names:
        # - shared base (data + 1)
        # - variable a (+10)
        # - variable b (+20)
        # Independent fusion would only have 2 (base embedded in each)
        assert len(key_names) == 3, (
            f"Expected 3 unique key names (shared base + 2 vars), got {len(key_names)}"
        )

    def test_fusion_preserves_shared_computation_at_runtime(self):
        """Verify shared computation is only executed once per chunk."""
        call_count = [0]

        def tracked_op(x):
            call_count[0] += 1
            return x + 1 + 2

        # Create shared base that branches to two variables
        data = da.ones((10, 10), chunks=5)  # 4 chunks
        base = data.map_blocks(tracked_op, dtype=float)

        ds = xr.Dataset(
            {
                "a": (["x", "y"], base + 10),
                "b": (["x", "y"], base + 20),
            }
        )

        # Verify fusion still reduces tasks
        expr = ds.expr
        fused = expr.fuse()
        assert len(fused.__dask_graph__()) < len(expr.__dask_graph__())

        # Compute and verify shared base is computed only once per chunk
        call_count[0] = 0
        result = ds.compute()

        # base has 4 chunks, should be computed exactly 4 times (not 8)
        assert call_count[0] == 4, (
            f"Shared base computed {call_count[0]} times, expected 4"
        )

        # Verify results are correct
        expected_base = np.ones((10, 10)) + 1 + 2
        xr.testing.assert_equal(
            result["a"],
            xr.DataArray(expected_base + 10, dims=["x", "y"]),
        )
        xr.testing.assert_equal(
            result["b"],
            xr.DataArray(expected_base + 20, dims=["x", "y"]),
        )

    def test_optimization_called_during_compute(self):
        """Verify that optimize() is called automatically during dask.compute().

        This test verifies that when calling dask.compute(ds), the expression
        pipeline runs optimize() which includes fusion. We check this by
        comparing the graph that would be executed to the optimized graph.
        """
        from unittest.mock import patch

        # Create dataset with chain of fusable operations
        data = da.ones((10, 10), chunks=5)
        ds = xr.Dataset({"a": (["x", "y"], data)})
        ds = ds + 1 + 2 + 3 + 4 + 5  # 5 additions - should fuse

        expr = ds.expr

        # Get task counts before and after optimization
        unoptimized_tasks = len(expr.__dask_graph__())
        optimized_expr = expr.optimize(fuse=True)
        optimized_tasks = len(optimized_expr.__dask_graph__())

        # Sanity check: optimization should reduce tasks
        assert optimized_tasks < unoptimized_tasks, (
            f"Expected optimization to reduce tasks: {unoptimized_tasks} -> {optimized_tasks}"
        )

        # Patch optimize() to track if it's called
        from dask._expr import Expr

        optimize_called = [False]
        original_optimize = Expr.optimize

        def tracking_optimize(self, fuse=False):
            optimize_called[0] = True
            return original_optimize(self, fuse=fuse)

        with patch.object(Expr, "optimize", tracking_optimize):
            (result,) = dask.compute(ds)

        # Verify optimize was called
        assert optimize_called[0], "optimize() was not called during dask.compute()"

        # Verify result is correct
        expected = np.ones((10, 10)) + 15
        np.testing.assert_array_equal(result["a"].values, expected)

    def test_reshape_mean_slice_workflow(self):
        """Test workflow with reshape, mean, and slice.

        This tests that high-level expressions like Reshape are properly
        lowered through the optimization pipeline.
        """
        from dask._expr import FinalizeCompute
        from dask.base import collections_to_expr

        # Create data with shape that can be reshaped
        data = da.ones((100, 20, 20), chunks=(10, 10, 10))
        ds = xr.Dataset(
            {"temp": (["time", "lat", "lon"], data)},
            coords={
                "time": np.arange(100),
                "lat": np.arange(20),
                "lon": np.arange(20),
            },
        )

        # Apply operations: mean then slice
        daily = ds.mean(dim=["time"])
        subset = daily.isel(lat=slice(5, 10), lon=slice(5, 10))

        # Verify compute works through the full pipeline
        coll_expr = collections_to_expr([subset])
        final_expr = FinalizeCompute(coll_expr)
        optimized = final_expr.optimize()

        # Should be able to build graph without npartitions error
        graph = optimized.__dask_graph__()
        assert len(graph) > 0

        # Verify slice is pushed down by checking leaf shapes
        inner_expr = optimized.operands[0].var_exprs[0]

        def get_leaves(e, seen=None):
            if seen is None:
                seen = set()
            if e._name in seen:
                return []
            seen.add(e._name)
            deps = e.dependencies() if hasattr(e, "dependencies") else []
            if not deps:
                return [e.shape if hasattr(e, "shape") else None]
            leaves = []
            for d in deps:
                leaves.extend(get_leaves(d, seen))
            return leaves

        leaf_shapes = [s for s in get_leaves(inner_expr) if s is not None]
        # At least one leaf should have the sliced shape (5x5), not full (20x20)
        has_sliced = any(5 in s for s in leaf_shapes if isinstance(s, tuple))
        assert has_sliced, f"Slice not pushed down. Leaf shapes: {leaf_shapes}"

        # Verify compute produces correct result
        result = subset.compute()
        assert result["temp"].shape == (5, 5)  # lat=5, lon=5
