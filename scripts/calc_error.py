#%%
from mats_l1b_tools import read_functions as rf
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

#%%

def add_stat_error(ds_slice,channel):   
    ds_slice = photons2lsb(ds_slice,channel,add_to_ds = True)
    ds_slice = calc_bit_window(ds_slice)
    shot_noise = calc_shot_noise(ds_slice,channel)
    digi_noise = calc_digitization_noise(ds_slice,channel)
    compression_noise = calc_compression_noise(ds_slice,channel)

    ds_slice['StatisticalError'] = np.sqrt(shot_noise**2 + digi_noise**2 + compression_noise**2)

    return ds_slice

def calc_shot_noise(ds_slice,channel):
    if not ('ImageCalibrated_lsb' in ds_slice.variables):
        raise KeyError('Image in LSB not avalible')
    alpha = data[channel]['alpha'] #gain (electron per count)
    amp_correction = data[channel]['amp_correction'] #correction for pre-amplificaion (UV channels only)
 
    ImageCalibrated_electrons = ds_slice.ImageCalibrated_lsb*alpha/amp_correction
    error_electrons = np.sqrt(ImageCalibrated_electrons)
    error_lsb = error_electrons/alpha*amp_correction
    error_photons = lsb2photons(ds_slice, channel, error_lsb)

    return error_photons

def calc_digitization_noise(ds_slice,channel):
    if not ('WindowMode' in ds_slice.variables):
        raise KeyError('WindowMode not avalible')
    data_window = ds_slice['WindowMode']
    noise_lsb = 2**data_window
    noise_ph = lsb2photons(ds_slice,channel,noise_lsb)
    return noise_ph

def calc_compression_noise(dc_slice,channel):
    if not ('WindowMode' in ds_slice.variables):
        raise KeyError('WindowMode not avalible')
    data_window = ds_slice['WindowMode']
    jpeq = ds_slice.JPEGQ
    #takes in CCDitem and calulates shot noise
    rms_noise = np.array([6.2,6,5.3,4.5,3.3,2,0]) #from MATS CDR (can probably be remade)
    jpegq = np.array([70,75,80,85,90,95,100]) #from MATS CDR (can probably be remade)
    jpeg_noise = np.interp(jpeq,jpegq,rms_noise)
    noise_lsb = jpeg_noise*(2**data_window)
    noise_ph = lsb2photons(ds_slice,channel,noise_lsb)

    return noise_ph

def photons2lsb(ds_slice, channel, add_to_ds = False):
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

    if add_to_ds:
        ds_slice['ImageCalibrated_lsb'] = ImageCalibrated_lsb
        return ds_slice
    else:
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

def bits_needed(arr):
    """
    Calculate the bit length for each element in a numeric array, handling non-integer types and errors.
    For non-integer types, elements are converted to integers. If the bit length exceeds 12, it is reduced by 12,
    otherwise, it is set to 0. If conversion fails, the position is filled with NaN.

    Args:
        arr (np.ndarray): A numpy array of numbers for which to calculate bit lengths.

    Returns:
        np.ndarray: An array of the same shape as `arr` containing the calculated bit lengths or modified bit lengths.
    """
    output = np.full(arr.shape, np.nan)  # Initialize an output array of the same shape filled with NaN
    
    for idx, element in np.ndenumerate(arr):
        try:
            if isinstance(element, (int, np.integer)):
                output[idx] = element.bit_length()
            else:
                element = int(element)  # Convert to integer if it's a float or any float-like type
                output[idx] = np.maximum((element.bit_length() - 12), 0)
        except (ValueError, TypeError):
            continue  # Leave the output as NaN in case of an error
    
    return output 


def calc_bit_window(ds_slice):
    """
    Applies the bits_needed function to the maximum value of 'ImageCalibrated_lsb' across specified axes in an xarray dataset,
    and stores the result in a new variable 'WindowMode' within the dataset.

    Args:
        ds_slice (xarray.Dataset): The dataset containing 'ImageCalibrated_lsb' data array.

    Returns:
        xarray.Dataset: The updated dataset including a new 'WindowMode' data variable.
    """
    num_bits = xr.apply_ufunc(
        bits_needed, 
        ds_slice.ImageCalibrated_lsb.max(axis=1).max(axis=1)
    )
    ds_slice['WindowMode'] = num_bits

    return ds_slice

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
plt.title('Total noise IR1')
plt.show()
# %%
