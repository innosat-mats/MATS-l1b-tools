'''
Example to load local Xarray (from zarr) filter on latitudes and plot with tangent point altitudes
'''

#%%
import xarray as xr
from matplotlib import pyplot as plt
from mats_l1b_tools.geolocation import addTPdata,grid_on_tanalt
from mats_l1b_tools import fetch_data
import numpy as np
import os

#%% Fetches local MATS Xarray
package_root = os.path.dirname(os.path.dirname(__file__))
data_file_path = os.path.join(package_root, 'data', 'L1b_IR1-2023_3_1_0_0_0-2023_3_1_1_0_0.zarr')
ds = xr.open_zarr(data_file_path)

#Slice data to only include when Tangent latitude (center pixel) is between 10-30 
ds_slice = fetch_data.filter(ds,[['TPlat',10,30]])

#Add geolocation of each pixel to image
ds_slice = addTPdata(ds_slice)

#%%
# Plot the 50th image adding tangent altitude info
plt.figure()
plt.imshow(ds_slice.ImageCalibrated.isel(time=50).values,origin='lower',aspect='auto')
plt.colorbar()
tanalt_data = ds_slice.isel(time=50).TPheightPixel*1e-3
contour_plot = plt.contour(tanalt_data, colors='black', linestyles='dashed', origin='lower')
plt.clabel(contour_plot, inline=True, fontsize=8, fmt='%1.1f')
plt.xlabel('Image Column Number')
plt.ylabel('Image Row Number')
plt.title('Image from IR1 with tangent altitudes (km)')
plt.show()
# %%
# Plot the 50th image using tangent altitude as y axis

images_gridded,tanalt_grid = grid_on_tanalt(ds_slice)
#%%

plt.figure()
plt.pcolor(ds_slice.im_col.values,tanalt_grid*1e-3,images_gridded[:,:,50])
plt.xlabel('Image Column Number')
plt.ylabel('Tangent alitude (km)')
plt.colorbar()
plt.title('Image from IR1 aligned on tangent alts.')
plt.show()