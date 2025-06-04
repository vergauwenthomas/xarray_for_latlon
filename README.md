# xarray_for_latlon
Extension to xarray specific for features relying on lat lan coordinates



## Installation

For useage install it as a package:

```shell

pip install git+https://github.com/vergauwenthomas/xarray_for_latlon.git

```

If you want to contribute, clone the repo.


## Usage

Note: No PROJ or any projection info is required. The problem is solvend, as a traditional minimizing problem (in haversine space).

Example:

```python
import xarray as xr
from xarray_for_latlon import sel_nearest_latlon
# Example DataArray with lat/lon coordinates
da = xr.DataArray(
    [[1, 2], [3, 4]],
    coords={"lat": [10, 20], "lon": [30, 40]},
    dims=["lat", "lon"]
)

# Subset too the nearest point to a given lat/lon
da_at_point = sel_nearest_latlon(
                xrobj=da,
                indexers_equivalent={
                    'lat': 17,
                    'lon': 32,}
                )
print(result)
```