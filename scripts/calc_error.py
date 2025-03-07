#%%
from mats_l1b_tools import read_functions as rf
from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
import json
from scipy.io import loadmat

#%%
ds_slice = xr.open_zarr('../ir1.zarr')

with open('../data/calibration_data.json', 'r') as file:
    data = json.load(file)


#%%

def add_stat_error(ds_slice,channel):

    shot_noise = calc_shot_noise(ds_slice,channel)
    ds_slice['StatisticalError'] = shot_noise

    return ds_slice

def calc_shot_noise(dc_slice,channel):
    alpha = data[channel]['alpha'] #gain (electron per count)
    amp_correction = data[channel]['amp_correction'] #correction for pre-amplificaion (UV channels only)

    ImageCalibrated_lsb = photons2lsb(dc_slice,channel)
    ImageCalibrated_electrons = ImageCalibrated_lsb*alpha/amp_correction
    error_electrons = np.sqrt(ImageCalibrated_electrons)
    error_lsb = error_electrons/alpha*amp_correction
    error_photons = lsb2photons(ds_slice, channel, error_lsb)

    return error_photons

def photons2lsb(ds_slice, channel):
    """ Converts photon counts to least significant bits (LSBs) using calibration data.

    Args:
        ds_slice: xarray.Dataset containing photon counts and calibration metadata.
        channel: String identifier for the channel to access specific calibration constants.

    Returns:
        xarray.DataArray: Calibrated image data in LSBs, accounting for exposure time and binning.
    """
    totbin = int(ds_slice["NRBIN"]) * int(ds_slice["NCBINCCDColumns"]) * int(ds_slice["NCBINFPGAColumns"])
    texp_seconds = ds_slice['TEXPMS'].item() / 1000  # Convert ms to seconds
    absolute_calibration = data[channel]['absolute_calibration']  # calibration constant

    # Perform the calculation
    ImageCalibrated_lsb = ds_slice['ImageCalibrated'] * texp_seconds * absolute_calibration * totbin  # counts in each binned pixel

    return ImageCalibrated_lsb

def lsb2photons(ds_slice, channel, ImageCalibrated_lsb):
    """ Converts calibrated image data from least significant bits (LSBs) back to photon counts.

    Args:
        ds_slice: xarray.Dataset containing LSB data and calibration metadata.
        channel: String identifier for the channel to access specific calibration constants.
        ImageCalibrated_lsb: xarray.DataArray containing the LSB data to be converted back to photon counts.

    Returns:
        xarray.DataArray: Original photon counts data, reversing the calibration accounting for exposure time and binning.
    """
    totbin = int(ds_slice["NRBIN"]) * int(ds_slice["NCBINCCDColumns"]) * int(ds_slice["NCBINFPGAColumns"])
    texp_seconds = ds_slice['TEXPMS'].item() / 1000  # Convert ms to seconds
    absolute_calibration = data[channel]['absolute_calibration']  # calibration constant

    # Perform the inverse calculation
    ImageCalibrated = ImageCalibrated_lsb / (texp_seconds * absolute_calibration * totbin)  # Reverse to photon counts

    return ImageCalibrated



#%% only working for IR1 at the moment
channel = 'IR1'

#%% Add error to Xarray
ds_slice = add_stat_error(ds_slice,channel)# %%

# %% Calculate an plot error ratio
error_ratio = (ds_slice['StatisticalError'] / ds_slice['ImageCalibrated'])
mean_errors = error_ratio.mean(axis=2).to_numpy()*100
# Plotting
fig, ax = plt.subplots()
ax.plot(mean_errors.T[:,::100],np.arange(0,mean_errors.shape[1]))  # Multiply by 100 to convert to percentage
ax.set_xlim([0, 20])  # Set limits for x-axis
ax.set_xlabel('Error %')  # Set x-axis label
ax.set_ylabel('Row number')
plt.title('Shot noise IR1')
plt.show()