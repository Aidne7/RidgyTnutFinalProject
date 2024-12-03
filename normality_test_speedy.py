import numpy as np
from netCDF4 import Dataset
from scipy.stats import shapiro
import pickle
import sys
import os
from datetime import datetime, timedelta

# Parse command-line arguments
if len(sys.argv) != 5:
    print("Usage: python normality_test_speedy.py <days_since_20110101> <ensemble_name> <variable_name> <output_dir>")
    sys.exit(1)

days_since_20110101 = int(sys.argv[1])  # Number of days since 1 Jan 2011
ensemble_name = sys.argv[2]  # e.g., reference_ens or perturbed_ens
variable_name = sys.argv[3]  # e.g., u, v, t
output_dir = sys.argv[4]  # Directory to save pickle files

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Compute the date in ccyymmddhh format
date = (datetime(2011, 1, 1, 0, 0) + timedelta(days=days_since_20110101)).strftime('%Y%m%d%H')

# Path to NetCDF file
file_path = f"/fs/ess/PAS2856/SPEEDY_ensemble_data/{ensemble_name}/{date}00.nc"

# Load data using netCDF4

ds = Dataset(file_path, 'r')
data = ds.variables[variable_name][:]  # Load the variable (assumes float array)
sigma = ds.variables['lev'][:]  # Model sigma levels


# Debugging: Print shape of data
print("Data shape:", data.shape)


# Adjust for time dimension if present
if len(data.shape) == 5:  # e.g., (time, levels, lat, lon, ensemble)
    num_time, num_levels, num_lat, num_lon, num_ens = data.shape
    data = data[0, :, :, :, :]  # Select first time step
elif len(data.shape) == 4:  # Expected shape
    num_levels, num_lat, num_lon, num_ens = data.shape

# Initialize an empty array for p-values
p_values = np.empty((num_levels, num_lat, num_lon))

# Loop through grid points and calculate p-values
for k in range(num_levels):
    for j in range(num_lat):
        for i in range(num_lon):
            ensemble_values = data[k, j, i, :]
            
            # Perform Shapiro-Wilk test and extract p-value
            _, p_value = shapiro(ensemble_values)

            p_values[k, j, i] = p_value

# Compute theoretical pressure
theoretical_pressure = sigma * 1000  # Assuming surface pressure is 1000 hPa

# Prepare dictionary for saving
result_dict = {
    "date": date,
    "vname": variable_name,
    "pvalues": p_values,
    "theoretical_pressure": theoretical_pressure
}

# Create filename and save pickle
pickle_filename = f"{variable_name}_{ensemble_name}_{date}00_pvalues.pkl"
pickle_path = os.path.join(output_dir, pickle_filename)

with open(pickle_path, 'wb') as f:
    pickle.dump(result_dict, f)

print(f"Results saved to {pickle_path}")
