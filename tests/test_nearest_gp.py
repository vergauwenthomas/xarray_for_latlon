import pytest
import numpy as np
import xarray as xr
import os, sys
from pathlib import Path
repo_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(repo_dir))
from xarray_for_latlon.nearest_gp import sel_nearest_latlon


@pytest.fixture
def custom_dataset():
    # Create a simple 3x3 grid with lat/lon
    lats = np.array([[10, 10, 10], [20, 20, 20], [30, 30, 30]])
    lons = np.array([[100, 110, 120], [100, 110, 120], [100, 110, 120]])
    data = np.arange(9).reshape(3, 3)
    ds = xr.Dataset(
        {
            "var": (("x", "y"), data)
        },
        coords={
            "lat": (("x", "y"), lats),
            "lon": (("x", "y"), lons),
            "x": np.arange(3),
            "y": np.arange(3)
        }
    )
    ds = ds.set_index(x=("x",), y=("y",))
    return ds

@pytest.fixture
def custom_dataarray():
    # Same as above, but as DataArray
    lats = np.array([[10, 10, 10], [20, 20, 20], [30, 30, 30]])
    lons = np.array([[100, 110, 120], [100, 110, 120], [100, 110, 120]])
    data = np.arange(9).reshape(3, 3)
    da = xr.DataArray(
        data,
        dims=("x", "y"),
        coords={
            "lat": (("x", "y"), lats),
            "lon": (("x", "y"), lons),
            "x": np.arange(3),
            "y": np.arange(3)
        }
    )
    da = da.set_index(x=("x",), y=("y",))
    return da

def test_sel_nearest_latlon_dataset(custom_dataset):
    # Target near (21, 111)
    result = sel_nearest_latlon(
        custom_dataset,
        {"lat": 21, "lon": 111}
    )
    # Should select the closest grid point (20, 110)
    assert np.isclose(result["lat"].item(), 20)
    assert np.isclose(result["lon"].item(), 110)
    assert result["var"].item() == 4

def test_sel_nearest_latlon_dataarray(custom_dataarray):
    # Target near (29, 119)
    result = sel_nearest_latlon(
        custom_dataarray,
        {"lat": 29, "lon": 119}
    )
    # Should select the closest grid point (30, 120)
    assert np.isclose(result["lat"].item(), 30)
    assert np.isclose(result["lon"].item(), 120)
    assert result.item() == 8

def test_sel_nearest_latlon_with_testdata():
    # Assume testdata/test_grid.nc exists with lat/lon/x/y
    testdata_path = os.path.join(
        os.path.dirname(__file__), "testdata", "ICMSHCSMK+0017h00m00s.nc"
    )
    ds = xr.open_dataset(testdata_path)
    # Pick a target near the center of the grid
    lat0 = 51.2355
    lon0 = 4.67
    result = sel_nearest_latlon(
        ds,
        {"lat": lat0, "lon": lon0}
    )
    # Should return a single point
    assert int(result['y'].data) == 219
    assert int(result['x'].data) == 212

    assert result["lat"].size == 1
    assert result["lon"].size == 1

def test_sel_nearest_latlon_with_testdata_time_concat():
    # Assume testdata/test_grid.nc exists with lat/lon/x/y
    datafolder = Path(os.path.join(os.path.dirname(__file__), "testdata"))
    ds = xr.open_mfdataset(datafolder.glob('ICMSHCSMK+*.nc'))

    # Pick a target near the center of the grid
    lat0 = 51.2355
    lon0 = 4.67
    result = sel_nearest_latlon(
        ds,
        {"lat": lat0, "lon": lon0}
    )
    # Should return a single point
    assert int(result['y'].data) == 219
    assert int(result['x'].data) == 212

    assert result["lat"].size == 1
    assert result["lon"].size == 1