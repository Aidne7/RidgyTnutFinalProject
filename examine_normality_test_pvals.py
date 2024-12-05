import argparse
import os
import pickle
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import false_discovery_control

#/users/PAS2856/ridgway72/.conda/envs/as4194/bin/python /users/PAS2856/ridgway72/FinalProjectLocal/RidgyTnutFinalProject/examine_normality_test_pvals.py --start_date 2011-01-02 --end_date 2011-04-10 --variable u --ensemble_type reference


# Argument parser
parser = argparse.ArgumentParser(description="Examine Normality Test P-Values with FDR Control")
parser.add_argument("--start_date", type=str, required=True, help="Start date (YYYY-MM-DD)")
parser.add_argument("--end_date", type=str, required=True, help="End date (YYYY-MM-DD)")
parser.add_argument("--variable", type=str, required=True, help="Variable to analyze")
parser.add_argument("--ensemble_type", type=str, choices=["reference", "perturbed"], required=True)
parser.add_argument("--interval", type=int, default=1, help="Interval in days between pickle files")

args = parser.parse_args()

# Generate date range
start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
dates = [start_date + timedelta(days=i) for i in range(0, (end_date - start_date).days + 1, args.interval)]

# Load pickle files
pval_arrays = []
for date in dates:
    file_path = os.path.join(
        "/fs/scratch/PAS2856/AS4194_Project/RidgewayNutting",
        f"{args.variable}_{args.ensemble_type}_ens_{date.strftime('%Y%m%d')}0000_pvalues.pkl"
    )
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            # Extract the p-values assuming they are under a key like 'pvalues'
            if 'pvalues' in data:
                pval_arrays.append(data['pvalues'])

# Combine into a 4D array
pvals_4d = np.stack(pval_arrays)  # Shape: (time, 8, 48, 96)

# Flatten for FDR correction
pvals_flat = pvals_4d.ravel()


# Perform BY correction
adjusted_pvals = false_discovery_control(pvals_flat, method="by")

# Create a boolean array indicating whether each p-value is rejected at alpha = 0.05
alpha = 0.05
rejected = adjusted_pvals < alpha

# Reshape results back to 4D
adjusted_pvals_4d = adjusted_pvals.reshape(pvals_4d.shape)
rejected_4d = rejected.reshape(pvals_4d.shape)

# Output results
print("Adjusted p-values shape:", adjusted_pvals_4d.shape)
print("Rejected null hypothesis shape:", rejected_4d.shape)





#Part C

import matplotlib.pyplot as plt
import seaborn as sns

# Plotting
# (i) Variation with Latitude
rejections_by_latitude = np.sum(rejected_4d, axis=(0, 1))  #Sum over time and levels
plt.figure(figsize=(10, 6))
plt.plot(rejections_by_latitude, label="Rejections by Latitude")
plt.xlabel("Latitude Index")
plt.ylabel("Number of Rejections")
plt.title("Null Hypothesis Rejections by Latitude")
plt.legend()
plt.savefig('Rejection with Latitude')

# (ii) Variation with Model Levels
rejections_by_level = np.sum(rejected_4d, axis=(0, 2))  #Sum over time and latitude
plt.figure(figsize=(10, 6))
plt.plot(rejections_by_level, label="Rejections by Model Level")
plt.xlabel("Model Level Index")
plt.ylabel("Number of Rejections")
plt.title("Null Hypothesis Rejections by Model Level")
plt.legend()
plt.savefig('Rejection vs Model Levels')

# (iii) Variation with Time (You will probably have to reformat this to datetime format)
rejections_by_time = np.sum(rejected_4d, axis=(1, 2))  #Sum over latitude and levels
plt.figure(figsize=(10, 6))
plt.plot(rejections_by_time, label="Rejections by Time")
plt.xlabel("Time Index")
plt.ylabel("Number of Rejections")
plt.title("Null Hypothesis Rejections Over Time")
plt.legend()
plt.savefig('Rejection vs Time')

# Heatmap: Latitude vs Time
heatmap_data = np.sum(rejected_4d, axis=1)  # Sum over model levels
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, cmap="viridis", cbar_kws={'label': 'Rejections'})
plt.xlabel("Latitude Index")
plt.ylabel("Time Index")
plt.title("Heatmap of Null Hypothesis Rejections (Latitude vs Time)")
plt.savefig('Latitude vs time')

#answer the questions 
"""
(i) For each model variable in the perturbed ensemble, how does the number of null
hypothesis rejections (i.e., non-Gaussian data) vary with latitude, model level and
time?


(ii) For each model variable, do the patterns in the perturbed ensemble’s null
hypothesis rejections become visually indistinguishable from those of the reference
ensemble? If yes, when does that happen?

"""
=======
