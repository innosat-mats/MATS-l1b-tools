import numpy as np
from scipy.interpolate import interpn


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
    geoloc = np.transpose(ds["geoloc"].to_numpy(), axes=(2, 1, 3, 0))
    outpoints = np.transpose(np.stack(np.meshgrid(rows, cols, indexing='ij')),
                             axes=(1, 2, 0)).reshape(-1, 2)
    shape = (len(rows), len(cols), geoloc.shape[2], geoloc.shape[3])
    res = np.transpose(interpn((y, x), geoloc, outpoints, method='cubic').reshape(shape),
                       axes=(2, 3, 0, 1))

    # Add the interpolated data to the data set
    var = [{"name": "TPheightPixel", "long_name": "Altitude of tangent point for each pixel"},
           {"name": "TPECEFx", "long_name": "Tangent point x coordinate in ECEF frame, for each pixel"},
           {"name": "TPECEFy", "long_name": "Tangent point y coordinate in ECEF frame, for each pixel"},
           {"name": "TPECEFz", "long_name": "Tangent point z coordinate in ECEF frame, for each pixel"}]
    for i, v in enumerate(var):
        ds = ds.assign({v["name"]: (("time", "im_row", "im_col"), res[i, ...])})
        ds[v["name"]].attrs = {"long_name": v["long_name"], "units": "meter"}

    # Remove the variables that were used to store uninterpolated tangent point corrdinates
    ds = ds.drop_vars(["geo_data", "geoloc", "geo_coord_x", "geo_coord_y"])

    return ds

def grid_on_tanalt(ds,dz = 500):

    tp_min = ds.TPheightPixel.min()
    tp_max = ds.TPheightPixel.max()
    y_new = np.arange(tp_min,tp_max,dz)

    images_gridded = np.zeros([len(y_new),len(ds.im_col),len(ds.time)])
    for j in range(len(ds.time)):
        image_data = ds.ImageCalibrated.isel(time=50).values
        tanalt_data = ds.isel(time=50).TPheightPixel

        for i in range(image_data.shape[1]):
            images_gridded[:,i,j] = np.interp(y_new,tanalt_data[:,i],image_data[:,i])

    return images_gridded,y_new