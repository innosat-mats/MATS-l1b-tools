#%%
from mats_l1b_tools import error as error
from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
import json
from scipy.io import loadmat
import os
from mats_l1b_tools.fetch_data import fetch_MATS_l1b_data
import datetime

#get data
channel = 'IR1'
start_time = datetime.datetime(2023,3,1,0)
end_time = datetime.datetime(2023,3,1,1)
ds_slice = fetch_MATS_l1b_data(channel,start_time,end_time)

#%% Add error to Xarray
ds_slice = error.add_stat_error(ds_slice,channel)# %%
# %% Calculate an plot error ratio
error_ratio = (ds_slice['StatisticalError'] / ds_slice['ImageCalibrated'])
mean_errors = error_ratio.mean(axis=2).to_numpy()*100 # Multiply by 100 to convert to percentage
# Plotting
fig, ax = plt.subplots()
ax.plot(mean_errors.T[:,::100],np.arange(0,mean_errors.shape[1]))  #plot every 100th profile
ax.set_xlim([0, 20])  # Set limits for x-axis
ax.set_xlabel('Error %')  # Set x-axis label
ax.set_ylabel('Row number')
plt.title('Total noise IR1')
plt.show()
# %%
