'''
Example to fetch MATS data headers then download and plot image of interest
'''


#%%
import s3fs
import xarray as xr
from matplotlib import pyplot as plt
from mats_l1b_tools.geolocation import addTPdata
from mats_l1b_tools.fetch_data import fetch_MATS_l1b_data

#%% Fetches info for MATS data
ds = fetch_MATS_l1b_data('IR2')

# %%
# Plot the 1000th image using Xarrays built in plotting fuction
ds.ImageCalibrated.isel(time=1000).plot()


#%%
# Plot the 1000th image using matplotlib
plt.figure()
plt.imshow(ds.ImageCalibrated.isel(time=1000).values,origin='lower',aspect='auto')
plt.xlabel('Image Column Number')
plt.ylabel('Image Row Number')
plt.colorbar()
plt.title('Image from IR2')
plt.show()