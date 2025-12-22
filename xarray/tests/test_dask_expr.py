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
        """Verify optimization when computing Dataset alongside raw dask arrays."""
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

        # Compute together - both should be optimized together
        _result_ds, _result_arr = dask.compute(ds, arr)

        # base has 4 chunks, should only compute once total
        assert call_count[0] == 4, f"Expected 4, got {call_count[0]}"
