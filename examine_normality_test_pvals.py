import argparse
import os
import pickle
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import false_discovery_control

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
    file_name = f"{args.variable}_{args.ensemble_type}_{date.strftime('%Y%m%d')}.pkl"
    if os.path.exists(file_name):
        with open(file_name, "rb") as f:
            pval_arrays.append(pickle.load(f))
    else:
        print(f"File not found: {file_name}")

# Combine into a 4D array
if len(pval_arrays) == 0:
    raise ValueError("No p-value data loaded!")
pvals_4d = np.stack(pval_arrays)  # Shape: (time, 8, 48, 96)

# Flatten for FDR correction
pvals_flat = pvals_4d.ravel()

# Perform BY correction
adjusted_pvals, rejected = false_discovery_control(pvals_flat, method="by", alpha=0.05)

# Reshape results back to 4D
adjusted_pvals_4d = adjusted_pvals.reshape(pvals_4d.shape)
rejected_4d = rejected.reshape(pvals_4d.shape)

# Output results
print("Adjusted p-values shape:", adjusted_pvals_4d.shape)
print("Rejected null hypothesis shape:", rejected_4d.shape)

# Save results
output_file = f"fdr_results_{args.variable}_{args.ensemble_type}.npz"
np.savez(output_file, adjusted_pvals=adjusted_pvals_4d, rejected=rejected_4d)
print(f"Results saved to {output_file}")
