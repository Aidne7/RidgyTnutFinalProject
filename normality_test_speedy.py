import numpy as np
from netCDF4 import Dataset
from scipy.stats import shapiro
import pickle
import sys
import os
from datetime import datetime, timedelta

# Validate command-line arguments
if len(sys.argv) != 5:
    print("Usage: python normality_test_speedy.py <days_since_20110101> <ensemble_name> <variable_name> <output_dir>")
    sys.exit(1)

# Parse command-line arguments
days_since_20110101 = int(sys.argv[1])
ensemble_name = sys.argv[2]
variable_name = sys.argv[3]
output_dir = sys.argv[4]

# Compute date from the given number of days since January 1, 2011
date = (datetime(2011, 1, 1) + timedelta(days=days_since_20110101)).strftime('%Y%m%d%H%M')

# Define the file path for the input NetCDF file
file_path = f"/fs/ess/PAS2856/SPEEDY_ensemble_data/{ensemble_name}/{date}.nc"

# Load the NetCDF file and read the specified variable and model levels
ds = Dataset(file_path, 'r')
data = ds.variables[variable_name][:]
sigma = ds.variables['lev'][:]

# Print the shape of the original data array
print(f"Original data shape: {data.shape}")

# Reshape the data to focus on the first time step across all ensemble members
num_ens, num_time, num_levels, num_lat, num_lon = data.shape
data = data[:, 0, :, :, :]

# Print the updated shape of the data
print("Adjusted data shape:", data.shape)

# Initialize an array to store Shapiro-Wilk test p-values
p_values = np.empty((num_levels, num_lat, num_lon))

# Perform the Shapiro-Wilk test for normality at each grid point
for k in range(num_levels):
    for j in range(num_lat):
        for i in range(num_lon):
            ensemble_values = data[:, k, j, i]
            _, p_value = shapiro(ensemble_values)
            p_values[k, j, i] = p_value

# Print the shape of the p-values array
print("Shape of p-values array:", p_values.shape)

# Convert sigma levels to theoretical pressure (in Pa)
theoretical_pressure = sigma * 1000

# Save results to a dictionary
result_dict = {
    "date": date,
    "vname": variable_name,
    "pvalues": p_values,
    "theoretical_pressure": theoretical_pressure
}

# Write results to a pickle file
pickle_filename = f"{variable_name}_{ensemble_name}_{date}_pvalues.pkl"
pickle_path = os.path.join(output_dir, pickle_filename)

with open(pickle_path, 'wb') as f:
    pickle.dump(result_dict, f)

print(f"Results saved to {pickle_path}")
