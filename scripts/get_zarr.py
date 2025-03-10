import datetime as DT
import numpy as np
import argparse
import xarray as xr
from scipy.interpolate import interpn


def get_args():
    """ Parses command line arguments using argparse.

    Returns:
        argparse object with arguments.
    """

    parser = argparse.ArgumentParser(description="Download MATS data from a zarr database",
                                     formatter_class=argparse.MetavarTypeHelpFormatter)
    parser.add_argument("address", type=str, help="Zarr archive adress")

    # Data filtering options
    parser.add_argument("-b", "--start_time", type=int, nargs=6,
                        help="Start time for data set (YYYY MM DD hh mm ss)")
    parser.add_argument("-e", "--stop_time", type=int, nargs=6,
                        help="Stop time for data set (YYYY MM DD hh mm ss)")
    parser.add_argument("-f", "--filter", action='append', type=str, nargs=3,
                        help="Filter data by values of given variable: <variable> <lower bound> <upper bound>")

    # Output options
    parser.add_argument("-n", "--ncdf_out", type=str, help="Output file name for ncdf output.")
    parser.add_argument("-z", "--zarr_out", type=str, help="Output file name for zarr output.")

    # Other options
    parser.add_argument("-t", "--tp_coords", action="store_true",
                        help="Interpolate tangent point coordinates for every pixel.")

    return parser.parse_args()


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

    # Interpolate tangent point coordinate data
    x, y = ds["geo_coord_x"].to_numpy(), ds["geo_coord_y"].to_numpy()
    cols, rows = ds["im_col"].to_numpy(), ds["im_row"].to_numpy()
    geoloc = np.transpose(ds["geoloc"].to_numpy(), axes=(1, 2, 3, 0))
    outpoints = np.array(np.meshgrid(rows, cols, indexing='ij')).T.reshape(-1, 2)
    shape = (len(rows), len(cols), geoloc.shape[2], geoloc.shape[3])
    res = np.transpose(interpn((y, x), geoloc, outpoints, method='cubic').reshape(shape), axes=(3, 0, 1, 2))

    # Add the interpolated data to the data set
    var = [{"name": "TPheightPixel", "long_name": "Altitude of tangent point for each pixel"},
           {"name": "TPECEFx", "long_name": "Tangent point x coordinate in ECEF frame, for each pixel"},
           {"name": "TPECEFy", "long_name": "Tangent point y coordinate in ECEF frame, for each pixel"},
           {"name": "TPECEFz", "long_name": "Tangent point z coordinate in ECEF frame, for each pixel"}]
    for i, v in enumerate(var):
        ds = ds.assign({v["name"]: (("time", "im_row", "im_col"), res[:, :, :, i])})
        ds[v["name"]].attrs = {"long_name": v["long_name"], "units": "meter"}

    # Remove the variables that were used to store uninterpolated tangent point corrdinates
    ds = ds.drop_vars(["geo_data", "geoloc", "geo_coord_x", "geo_coord_y"])

    return ds


def main():
    # Parse command line arguments and open the specified zarr data set
    args = get_args()
    dsf = xr.open_zarr(args.address, storage_options={"anon": True})

    # Apply time filtering, if requested
    if np.logical_xor(args.start_time is None, args.stop_time is None):
        raise ValueError("Please specify both --start_time (-b) and --stop_time (-e) or neither.")
    if args.start_time:
        start_time, stop_time = [DT.datetime(*date) for date in [args.start_time, args.stop_time]]
        dsf = dsf.sel({"time": slice(start_time, stop_time)})

    # Apply all other filters (that were specified using -f option), if any
    if args.filter:
        dsf = filter(dsf, args.filter)

    # Verify that the data set is still non-empty after all the filtering
    if dsf.sizes["time"] == 0:
        raise RuntimeError("There are no images in the time interval (that match filter criteria, if any)!")
    print(f"Downloading {dsf.sizes['time']} images...")

    # Interpolate tangent point coordinates for each pixel, if requested
    if args.tp_coords:
        dsf = addTPdata(dsf)

    # Write output to the specified files
    if args.zarr_out:
        dsf.to_zarr(args.zarr_out, mode="w")
    if args.ncdf_out:
        dsf.to_netcdf(args.ncdf_out)

    # Default netCDF output, if no output file names were specified
    if not (args.zarr_out or args.ncdf_out):
        name = "L1b"
        if "channel" in dsf.keys():
            name += f"_{str(dsf['channel'].data)}"
        if args.start_time:
            dates = ["_".join([str(x) for x in array]) for array in [args.start_time, args.stop_time]]
            name += f"-{'-'.join(dates)}"
        dsf.to_netcdf(f"{name}.nc")


if __name__ == "__main__":
    main()
