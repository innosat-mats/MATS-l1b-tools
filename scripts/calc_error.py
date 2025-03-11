#%%
from mats_l1b_tools import add_info as add_info
from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
import json
from scipy.io import loadmat
import os

# Change the current working directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
#%%
ds_slice = xr.open_zarr('../ir1.zarr')

with open('../data/calibration_data.json', 'r') as file:
   data = json.load(file)

dc_data = loadmat('../data/darkcurrent/FM016_CCD_DC_calibration.mat')

#%% only working for IR1 at the moment
channel = 'IR1'

#%% Add error to Xarray
ds_slice = add_info.add_stat_error(ds_slice,channel)# %%
ro_error = calc_readout_noise(ds_slice,channel)
# %% Calculate an plot error ratio
error_ratio = (ds_slice['StatisticalError'] / ds_slice['ImageCalibrated'])
mean_errors = error_ratio.mean(axis=2).to_numpy()*100
# Plotting
fig, ax = plt.subplots()
ax.plot(mean_errors.T[:,::100],np.arange(0,mean_errors.shape[1]))  # Multiply by 100 to convert to percentage
ax.set_xlim([0, 20])  # Set limits for x-axis
ax.set_xlabel('Error %')  # Set x-axis label
ax.set_ylabel('Row number')
plt.title('Total noise IR1')
plt.show()
# %%
