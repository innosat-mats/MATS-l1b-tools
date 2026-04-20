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
channel = 'UV2'
start_time = datetime.datetime(2023,2,1,0)
end_time = datetime.datetime(2023,6,1,1)
ds_slice = fetch_MATS_l1b_data(channel,start_time,end_time)

#%%
print(ds_slice)
# print all variables
print(ds_slice.data_vars)

#%% Add error to Xarray
ds_slice = error.add_flags(ds_slice)# %%


# %% Plot the error flags

plt.figure()
for flag in ['FlagNonlinearCorrection','FlagSaturatedPixel', 'FlagSingleEvent',
        'FlagHotPixel']:
    ds_slice[flag].mean(dim="time").plot()
    print(f"{flag}: Pixel with maximum mean value has value of {ds_slice[flag].mean(dim='time').max().values}. ")

    plt.xlabel('Image Column Number')
    plt.ylabel('Image Row Number')
    # set the title to the flag long name
    plt.title(channel +' '+ ds_slice[flag].attrs['long_name'])
    plt.show()

#%%
# loop through the flags and plot the average of each flag in a subplot
flag_names = [
        'FlagBadColumns',
        'FlagSingleEvent',
        'FlagHotPixel',
        'FlagNoHotPixel',
        'FlagNegativeBias',
        'FlagNonlinearCorrection',
        'FlagSaturatedPixel',
        'FlagDesmearNegative',
        'FlagDesmearNoAtmosphere',
        'FlagDarkCurrentNegative',
        'FlagExtremeTemperature',
        'FlagNoTemperature',
        'FlagFlatfieldNegative',
        'FlagFlatfieldLargeFactor'
    ]
fig, ax= plt.subplots(nrows=4, ncols=4, figsize=(15, 10))
for i, flag in enumerate(flag_names):
    ax[i // 4, i % 4].imshow(ds_slice[flag].mean(dim="time").values, origin='lower', aspect='auto')
    ax[i // 4, i % 4].set_title(f"{channel} {ds_slice[flag].attrs['long_name']}")
    ax[i // 4, i % 4].set_xlabel('Image Column Number')
    ax[i // 4, i % 4].set_ylabel('Image Row Number')
    # Set colorbar for each subplot
    plt.colorbar(ax[i // 4, i % 4].images[0], ax=ax[i // 4, i % 4], orientation='vertical')

plt.tight_layout()
plt.suptitle('Error invocation rates Flags for ' + channel, fontsize=16)
plt.subplots_adjust(top=0.9)  # Adjust the top to make room for the suptitle
plt.show()
# %% Plot the error flags


# %%
