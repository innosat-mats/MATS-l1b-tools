from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
import json
from scipy.io import loadmat
import os

# Determine the directory in which the current file resides
package_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
data_file_path = os.path.join(package_root, 'data', 'calibration_data.json')
with open(data_file_path, 'r') as file:
    data = json.load(file)


def add_stat_error(ds_slice, channel):
    """
    Calculates all statistical error sources and adds them to an xarray dataset.

    Args:
        ds_slice: xarray.Dataset containing the original data.
        channel: String identifier for the channel to access specific calibration constants.

    Returns:
        ds_slice: xarray.Dataset updated with a new data variable 'StatisticalError' that contains the calculated statistical errors.
    """

    photons2lsb(ds_slice, channel,append_ds = True)
    calc_bit_window(ds_slice)
    
    shot_noise = calc_shot_noise(ds_slice, channel)
    digi_noise = calc_digitization_noise(ds_slice, channel)
    jpeg_noise = calc_compression_noise(ds_slice, channel)
    ro_noise = calc_readout_noise(ds_slice, channel)

    
    ds_slice['StatisticalError'] = np.sqrt(shot_noise**2 + digi_noise**2 + jpeg_noise**2 + ro_noise**2)

    return ds_slice

def calc_shot_noise(ds_slice, channel):
    """
    Calculates the shot noise for image data in an xarray dataset.

    Shot noise is calculated based on the calibrated image data converted to electron counts, considering gain and amplification corrections.

    Args:
        ds_slice: xarray.Dataset containing the image data.
        channel: String identifier for the channel to access specific constants like gain ('alpha') and amplification correction ('amp_correction').

    Returns:
        error_photons: xarray.DataArray representing the shot noise in terms of radiance units.
    """
    if 'ImageCalibrated_lsb' not in ds_slice.variables:
        raise KeyError('ImageCalibrated_lsb not available')

    alpha = data[channel]['alpha']  # gain (electron per count)
    amp_correction = data[channel]['amp_correction']  # correction for pre-amplification (UV channels only)

    ImageCalibrated_electrons = ds_slice['ImageCalibrated_lsb'] * alpha / amp_correction
    error_electrons = np.sqrt(ImageCalibrated_electrons)
    error_lsb = error_electrons / alpha * amp_correction
    error_photons = lsb2photons(ds_slice, channel, error_lsb)

    return error_photons

def calc_digitization_noise(ds_slice, channel):
    """
    Calculate the digitization noise in photons for a given channel based on the data window size specified in the 'WindowMode' variable.

    Args:
        ds_slice (xarray.Dataset): The dataset containing the 'WindowMode' variable.
        channel (str): The channel for which to calculate the noise.

    Returns:
        float: The calculated noise in photons for the specified channel.

    Raises:
        KeyError: If 'WindowMode' is not available in the dataset.
    """
    if 'WindowMode' not in ds_slice.variables:
        raise KeyError('WindowMode not available')
    
    data_window = ds_slice['WindowMode']
    noise_lsb = 2**data_window
    noise_ph = lsb2photons(ds_slice, channel, noise_lsb)
    
    return noise_ph

def calc_compression_noise(ds_slice, channel):
    """
    Calculate the compression noise in photons for a given channel based on JPEG quality and the data window size specified in the 'WindowMode' variable.

    Args:
        ds_slice (xarray.Dataset): The dataset containing the 'WindowMode' and 'JPEGQ' variables.
        channel (str): The channel for which to calculate the noise.

    Returns:
        float: The calculated compression noise in photons for the specified channel.

    Raises:
        KeyError: If 'WindowMode' is not available in the dataset.
    """
    if 'WindowMode' not in ds_slice.variables:
        raise KeyError('WindowMode not available')
    
    data_window = ds_slice['WindowMode']
    jpegq = ds_slice.JPEGQ
    # Predefined noise levels and corresponding JPEG quality levels from MATS CDR
    rms_noise = np.array([6.2, 6, 5.3, 4.5, 3.3, 2, 0])  # from MATS CDR (can probably be remade)
    jpegq_levels = np.array([70, 75, 80, 85, 90, 95, 100])  # from MATS CDR (can probably be remade)
    jpeg_noise = np.interp(jpegq, jpegq_levels, rms_noise)
    noise_lsb = jpeg_noise * (2**data_window)
    noise_ph = lsb2photons(ds_slice, channel, noise_lsb)
    
    return noise_ph

def calc_readout_noise(ds_slice, channel):

    alpha = data[channel]['alpha']  # gain (electron per count)
    amp_correction = data[channel]['amp_correction']  # correction for pre-amplification (UV channels only)
    e_noise = data[channel]['ro_avr'] #readout noise in electrons

    noise_lsb = e_noise/alpha*amp_correction
    noise_ph = lsb2photons(ds_slice, channel, noise_lsb)
    
    return noise_ph

def photons2lsb(ds_slice, channel,append_ds = False):
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

    if append_ds:
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

def add_flags(ds_slice):
    """
    Returns the error flags as new data variables, according to the below

    CalibrationErrors 	16 bit number (expressed in decimal form) where bits are according to below:
    Bit 1 	Flag set to 1 across image if bad column is present in image
    Bit 2 	Flag to indicate that single event that has been corrected
    Bit 3 	Flag to indicate a hot pixel, which has been corrected
    Bit 4 	Flag to indicate that no hot pixel correction has been done to the image
    Bit 5 	Flag to indicate that negative values appeared after bias subtraction
    Bit 6 	Flag to indicate that nonlinear correction of more than 5% has been applied for pixel
    Bit 7 	Flag to indicate that pixel is saturated in summation well, read out register or single pixel
    Bit 8 	Flag to indicate that the desmear subtraction rendered negative result
    Bit 9 	Flag to indicate that the desmear subtraction could not estimate realistic atmospheric parameters needed for desmearing, and thus no desmearing has been done
    Bit 10 	Flag to indicate that the dark current subtraction rendered negative result
    Bit 11 	Flag for extreme temperature (30C)
    Bit 12 	Flag indicating no temperature reading and default temperature of -15 C has been used
    Bit 13 	Flag to indicate flatfield correction rendered negative value
    Bit 14 	Flag to indicate abnormally large (>5%) flatfield compensation factor
    Bit 15 	Flag not used
    Bit 16 	Flag not used


    Args:
        ds_slice: xarray.Dataset containing the original data.
        channel: String identifier for the channel to access specific calibration constants.

    Returns:
        ds_slice: xarray.Dataset updated with a new data variables - one for each flag, with a boolean value indicating the presence of the flag.
    """

    if 'CalibrationErrors' not in ds_slice.variables:
        raise KeyError('CalibrationErrors not available')
    calibration_errors = ds_slice['CalibrationErrors'].astype(np.uint16)
    # Create a new data variable for each flag
    ds_slice['FlagBadColumns'] = (calibration_errors & 0b0000000000000001) > 0
    ds_slice['FlagSingleEvent'] = (calibration_errors & 0b0000000000000010) > 0
    ds_slice['FlagHotPixel'] = (calibration_errors & 0b0000000000000100) > 0
    ds_slice['FlagNoHotPixel'] = (calibration_errors & 0b0000000000001000) > 0
    ds_slice['FlagNegativeBias'] = (calibration_errors & 0b0000000000010000) > 0
    ds_slice['FlagNonlinearCorrection'] = (calibration_errors & 0b0000000000100000) > 0
    ds_slice['FlagSaturatedPixel'] = (calibration_errors & 0b0000000001000000) > 0
    ds_slice['FlagDesmearNegative'] = (calibration_errors & 0b0000000010000000) > 0
    ds_slice['FlagDesmearNoAtmosphere'] = (calibration_errors & 0b0000000100000000) > 0
    ds_slice['FlagDarkCurrentNegative'] = (calibration_errors & 0b0000001000000000) > 0
    ds_slice['FlagExtremeTemperature'] = (calibration_errors & 0b0000010000000000) > 0
    ds_slice['FlagNoTemperature'] = (calibration_errors & 0b0000100000000000) > 0
    ds_slice['FlagFlatfieldNegative'] = (calibration_errors & 0b0001000000000000) > 0
    ds_slice['FlagFlatfieldLargeFactor'] = (calibration_errors & 0b0010000000000000) > 0
    # Note:

    # Flags 15 and 16 are not used, so we do not create variables for them
    # Remove the original CalibrationErrors variable
    ds_slice = ds_slice.drop_vars('CalibrationErrors')
    # Add the flags to the dataset
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
    for flag in flag_names:
        ds_slice[flag].attrs['long_name'] = f"Flag indicating presence of {flag.replace('Flag', '').replace('No', 'no ')}"
        ds_slice[flag].attrs['units'] = 'boolean'
        ds_slice[flag].attrs['description'] = f"This flag is set to True if the {flag.replace('Flag', '').replace('No', 'no ')} is present in the image."
    
    return ds_slice