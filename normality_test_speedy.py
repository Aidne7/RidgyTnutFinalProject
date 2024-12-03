import numpy as np
from netCDF4 import Dataset
from scipy.stats import shapiro
import pickle
import sys
import os
from datetime import datetime, timedelta

# Check command-line arguments for correct usage
if len(sys.argv) != 5:
    print("Usage: python normality_test_speedy.py <days_since_20110101> <ensemble_name> <variable_name> <output_dir>")
    sys.exit(1)

# Parse input arguments
days_since_20110101 = int(sys.argv[1])
ensemble_name = sys.argv[2]
variable_name = sys.argv[3]
output_dir = sys.argv[4]

# Calculate the date based on days since January 1, 2011
date = (datetime(2011, 1, 1, 0, 0) + timedelta(days=days_since_20110101)).strftime('%Y%m%d%H%M')

# Construct the path to the NetCDF file
file_path = f"/fs/ess/PAS2856/SPEEDY_ensemble_data/{ensemble_name}/{date}.nc"

# Open the NetCDF file and read the specified variable
ds = Dataset(file_path, 'r')
data = ds.variables[variable_name][:]  # Load the data as a float array
sigma = ds.variables['lev'][:]  # Load model sigma levels

# Display the original shape of the data
print(f"Original data shape: {data.shape}")

# Extract dimensions for easier handling
num_ens, num_time, num_levels, num_lat, num_lon = data.shape

# Select the first time step across all ensembles and levels
data = data[:, 0, :, :, :]  # Simplify the data to include only the first time index

# Confirm the new shape of the data
print("Adjusted data shape:", data.shape)

# Initialize an array to store p-values for each grid point
p_values = np.empty((num_levels, num_lat, num_lon))

# Iterate over each level, latitude, and longitude to compute p-values
for k in range(num_levels):
    for j in range(num_lat):
        for i in range(num_lon):
            ensemble_values = data[:, k, j, i]  # Extract ensemble values for the current grid point

            # Perform the Shapiro-Wilk test and store the p-value
            _, p_value = shapiro(ensemble_values)
            p_values[k, j, i] = p_value

# Confirm the shape of the p-values array
print("Shape of p-values array:", p_values.shape)

# Calculate theoretical pressure from sigma levels
theoretical_pressure = sigma * 1000  # Convert sigma levels to pressure (Pa)

# Create a dictionary to save the results
result_dict = {
    "date": date,
    "vname": variable_name,
    "pvalues": p_values,
    "theoretical_pressure": theoretical_pressure
}

# Generate the output filename and save the dictionary as a pickle file
pickle_filename = f"{variable_name}_{ensemble_name}_{date}_pvalues.pkl"
pickle_path = os.path.join(output_dir, pickle_filename)

with open(pickle_path, 'wb') as f:
    pickle.dump(result_dict, f)

print(f"Results saved to {pickle_path}")
