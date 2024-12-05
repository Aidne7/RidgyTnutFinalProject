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


