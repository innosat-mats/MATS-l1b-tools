import datetime as DT
import numpy as np
import argparse
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

def filter(ds, filter_args):
    """ Filters xarray dataset using provided filters.

    Args:
        ds: Original xarray dataset
        filter_args: list of lists, each sublist of the form [<variable name>, <lower bound>, <upper bound>]

    Returns:
        ds: Filtered xarray dataset.
    """

    for filt in filter_args:
        lims = [float(x) for x in filt[1:]]
        if filt[0] == "time":
            raise ValueError("Please use --start_time and --stop_time for time filtering.")
        elif filt[0] in ds.dims:
            ds = ds.sel({filt[0]: slice(lims[0], lims[1])})
        elif filt[0] in ds.keys():
            if ds[filt[0]].dims != ("time",):
                raise ValueError("Filters can only be created for dimensional variables and " +
                                 "variables with a single value per image.")
            ds = ds.where(np.logical_and(ds[filt[0]] > lims[0], ds[filt[0]] < lims[1]), drop=True)
        else:
            raise ValueError(f"The variable {filt[0]} is not in the data set!")
    return ds


def addTPdata(ds):
    """ Replaces low-resolution tangent point geolocation data in the original data set with values
        interpolated for every pixel of every image.

    Args:
        ds: Original xarray dataset

    Returns:
        ds: xarray dataset with interpolated tangent point location data.
    """

    # Creating new variables in xarray dataset for storing tangent point coordinates for each pixel in the image
    shape = tuple([ds.sizes[name] for name in ["time", "im_row", "im_col"]])
    tp_val_template = (("time", "im_row", "im_col"), np.zeros(shape))
    var = [{"name": "TPheightPixel", "long_name": "Altitude of tangent point for each pixel"},
           {"name": "TPECEFx", "long_name": "Tangent point x coordinate in ECEF frame, for each pixel"},
           {"name": "TPECEFy", "long_name": "Tangent point y coordinate in ECEF frame, for each pixel"},
           {"name": "TPECEFz", "long_name": "Tangent point z coordinate in ECEF frame, for each pixel"}]
    for v in var:
        ds = ds.assign({v["name"]: tp_val_template})
        ds[v["name"]].attrs = {"long_name": v["long_name"], "units": "meter"}

    # Interpolating tangent point coordinate data
    xx, yy = np.meshgrid(ds["im_col"][:], ds["im_row"][:], sparse=True)
    for im in range(shape[0]):
        for coord in range(len(var)):
            interpolator = RegularGridInterpolator([ds["geo_coord_x"].to_numpy(), ds["geo_coord_y"].to_numpy()],
                                                   ds["geoloc"][im, :, :, coord].to_numpy(), method='cubic')
            ds[var[coord]["name"]][im, :, :] = interpolator((xx, yy))

    # Removing variables that were used for storing sparse tangent point coordinates
    ds.drop_vars(["geo_data", "geoloc", "geo_coord_x", "geo_coord_y"])
    ds.drop_dims(["geo_coord_x", "geo_coord_y", "geo_data"])

    return ds


def read_data(address,start_time='2023 02 28 00 00 00',stop_time='2023 02 29 00 00 00',zarr_out=False,ncdf_out=False,filters=False,tp_coords=False):
    # Parse command line arguments and open the specified zarr data set
    args = get_args()
    dsf = xr.open_zarr(address, storage_options={"anon": True})

    # Apply time filtering, if requested
    if np.logical_xor(start_time is None, stop_time is None):
        raise ValueError("Please specify both --start_time (-b) and --stop_time (-e) or neither.")
    if args.start_time:
        start_time, stop_time = [DT.datetime(*date) for date in [start_time, stop_time]]
        ds_slice = dsf.sel({"time": slice(start_time, stop_time)})

    # Apply all other filters (that were specified using -f option), if any
    if filters:
        ds_slice = filter(ds_slice, filters)

    # Verify that the data set is still non-empty after all the filtering
    if ds_slice.sizes["time"] == 0:
        raise RuntimeError("There are no images in the time interval (that match filter criteria, if any)!")

    # Interpolate tangent point coordinates for each pixel, if requested
    if tp_coords:
        ds_slice = addTPdata(ds_slice)

    # Write output to the specified files
    print(f"Downloading {ds_slice.sizes['time']} images...")
    if zarr_out:
        ds_slice.to_zarr(zarr_out, mode="w")
    if ncdf_out:
        ds_slice.to_netcdf(ncdf_out)

    # Default netCDF output, if no output file names were specified just return slice

    return ds_slice