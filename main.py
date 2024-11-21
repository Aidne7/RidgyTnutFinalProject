print('test donkey test')

import numpy as np
from scipy.ndimage import gaussian_filter




def three_scale_decomposition(data2d):

    # Large scale: apply a strong Gaussian filter
    data_large_band_2d = gaussian_filter(data2d, sigma=10)
    
    # Medium scale: apply a less intense Gaussian filter and subtract large scale
    medium_scale = gaussian_filter(data2d, sigma=5)
    data_medium_band_2d = medium_scale - data_large_band_2d
    
    # Small scale: subtract medium scale from original data
    data_small_band_2d = data2d - medium_scale
    
    return data_large_band_2d, data_medium_band_2d, data_small_band_2d


def preprocess_geopotential(geo_data):
    """
    Prepares the geopotential ensemble data by removing the time dimension and 
    subsampling longitudes.
    
    Parameters:
        geo_data (np.ndarray): Input 4D array of shape (time, members, lat, lon).
    
    Returns:
        processed_data (np.ndarray): Output array of shape (1000, 8, 48, 48).
    """
    # Remove time dimension
    geo_data_no_time = geo_data[:, :, :, :]
    
    # Subsample longitudes (every other point)
    processed_data = geo_data_no_time[:, :, :, ::2]
    
    return processed_data


def apply_three_scale(data4d):
    """
    Applies three_scale_decomposition across all members in a 4D ensemble.
    
    Parameters:
        data4d (np.ndarray): Input 4D array of shape (1000, 8, 48, 48).
    
    Returns:
        large_scale (np.ndarray): Large-scale band array of shape (1000, 8, 48, 48).
        medium_scale (np.ndarray): Medium-scale band array of shape (1000, 8, 48, 48).
        small_scale (np.ndarray): Small-scale band array of shape (1000, 8, 48, 48).
    """
    large_scale, medium_scale, small_scale = [], [], []
    for i in range(data4d.shape[0]):  # Loop over time
        for j in range(data4d.shape[1]):  # Loop over members
            l, m, s = three_scale_decomposition(data4d[i, j])
            large_scale.append(l)
            medium_scale.append(m)
            small_scale.append(s)
    
    return (np.array(large_scale).reshape(data4d.shape),
            np.array(medium_scale).reshape(data4d.shape),
            np.array(small_scale).reshape(data4d.shape))

def compute_variance(scale_band_array):
    """
    Computes ensemble variance for a given scale band array.
    
    Parameters:
        scale_band_array (np.ndarray): 4D array of shape (1000, 8, 48, 48).
    
    Returns:
        variance (np.ndarray): 3D array of shape (8, 48, 48).
    """
    return np.var(scale_band_array, axis=0)

def compute_ensemble_3scale_variance(data4d):
    """
    Computes ensemble variance for large, medium, and small scale bands.
    
    Parameters:
        data4d (np.ndarray): Input ensemble data of shape (1000, 8, 48, 48).
    
    Returns:
        large_scale_variance (np.ndarray): Large scale band variance of shape (8, 48, 48).
        medium_scale_variance (np.ndarray): Medium scale band variance of shape (8, 48, 48).
        small_scale_variance (np.ndarray): Small scale band variance of shape (8, 48, 48).
    """
    large_scale, medium_scale, small_scale = apply_three_scale(data4d)
    return (compute_variance(large_scale),
            compute_variance(medium_scale),
            compute_variance(small_scale))


test_case = test