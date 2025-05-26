import datetime as DT
import numpy as np
import argparse
import xarray as xr
import s3fs

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

def datefilter(ds: xr.Dataset, start: DT.datetime, stop: DT.datetime) -> xr.Dataset:
    """
    Filter the dataset to include only data between the specified start and stop datetimes.

    Args:
    ds (xr.Dataset): The input dataset.
    start (DT.datetime): The start datetime.
    stop (DT.datetime): The stop datetime.

    Returns:
    xr.Dataset: The filtered dataset.
    """
    ds = ds.sel({"time": slice(start, stop)})
    return ds

def fetch_MATS_l1b_data(channel: str, start_time: DT.datetime = None, stop_time: DT.datetime = None) -> xr.Dataset:
    """
    Fetch MATS level 1b data from an S3 bucket for a given channel and optional time range.

    Args:
    channel (str): The data channel to fetch.
    start_time (DT.datetime, optional): The start datetime for data filtering.
    stop_time (DT.datetime, optional): The stop datetime for data filtering.

    Returns:
    xr.Dataset: The fetched dataset with optional time filtering applied.
    """
    # Establish connection to the S3 filesystem
    fs = s3fs.S3FileSystem(
        anon=True,
        client_kwargs={"endpoint_url": "https://bolin.su.se/data/s3/"},
        skip_instance_cache=True
    )

    # Access data mapper for the specified channel
    mapper = fs.get_mapper(f"data/mats-level-1b-limb-cropd-1.0/mats-level-1b-limb-cropd-{channel}.zarr")

    # Open the dataset using the mapper
    ds = xr.open_zarr(
        mapper,
        consolidated=True,
    )

    # Apply time filtering if both start_time and stop_time are provided
    if np.logical_xor(start_time is not None, stop_time is not None):
        raise ValueError("Please specify both --start_time and --stop_time or neither.")
    if start_time is not None and stop_time is not None:
        ds = datefilter(ds, start_time, stop_time)

    return ds