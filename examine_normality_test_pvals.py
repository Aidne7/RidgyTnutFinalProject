import argparse
import os
import pickle
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import false_discovery_control
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

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

# Output shapes for verification
print("Adjusted p-values shape:", adjusted_pvals_4d.shape)
print("Rejected null hypothesis shape:", rejected_4d.shape)

def create_plot(data, title, xlabel, ylabel, filename, x_extent, y_extent, y_ticks=None):
    """Generates a heatmap plot."""
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(
        data.T,
        aspect="auto",
        cmap="plasma",
        origin="lower",
        extent=[x_extent[0], x_extent[1], y_extent[0], y_extent[1]],
    )

    # Format x-axis with dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))  # Show major ticks every 5 days
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    plt.xticks(rotation=45)

    # Add labels and title
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    if y_ticks:
        ax.set_yticks(y_ticks)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Percentage of Null Hypothesis Rejections", fontsize=12)

    # Add gridlines for readability
    ax.grid(color='white', linestyle='--', linewidth=0.5)

    # Save and show plot
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Prepare data for the three plots
# (i) Null hypothesis rejections by average of latitude and model levels over time
rejections_time_avg_lat_level = np.sum(rejected_4d, axis=(1, 2))  # Sum over model levels and latitudes
total_lat_levels = rejected_4d.shape[1] * rejected_4d.shape[2]
percent_rejections_time_avg_lat_level = (rejections_time_avg_lat_level / total_lat_levels) * 100
time_steps = pd.date_range(start=args.start_date, end=args.end_date, freq=f"{args.interval}D")

create_plot(
    percent_rejections_time_avg_lat_level,
    f"Rejections by Time, Average Latitude, and Model Levels for {args.variable} ({args.ensemble_type} ensemble)",
    "Time",
    "Average Latitude & Model Levels",
    f"{args.variable}_{args.ensemble_type}_time_avg_lat_level.png",
    x_extent=[mdates.date2num(time_steps[0]), mdates.date2num(time_steps[-1])],
    y_extent=[0, 1],  # Set an appropriate y-range
)

# (ii) Null hypothesis rejections by latitude over time
rejections_time_lat = np.sum(rejected_4d, axis=(1, 2))  # Sum over model levels and longitudes
percent_rejections_time_lat = (rejections_time_lat / (rejected_4d.shape[2] * rejected_4d.shape[3])) * 100

create_plot(
    percent_rejections_time_lat,
    f"Rejections by Time and Latitude for {args.variable} ({args.ensemble_type} ensemble)",
    "Time",
    "Latitude Index",
    f"{args.variable}_{args.ensemble_type}_time_lat.png",
    x_extent=[mdates.date2num(time_steps[0]), mdates.date2num(time_steps[-1])],
    y_extent=[0, rejected_4d.shape[3]],
)

# (iii) Null hypothesis rejections by theoretical pressure using model levels over time
# Assuming model levels correspond to pressure levels; adapt accordingly if needed
rejections_time_pressure = np.sum(rejected_4d, axis=(2, 3))  # Sum over latitudes and longitudes
percent_rejections_time_pressure = (rejections_time_pressure / (rejected_4d.shape[2] * rejected_4d.shape[3])) * 100

create_plot(
    percent_rejections_time_pressure,
    f"Rejections by Time and Theoretical Pressure for {args.variable} ({args.ensemble_type} ensemble)",
    "Time",
    "Model Levels (Theoretical Pressure)",
    f"{args.variable}_{args.ensemble_type}_time_pressure.png",
    x_extent=[mdates.date2num(time_steps[0]), mdates.date2num(time_steps[-1])],
    y_extent=[0, rejected_4d.shape[1]],
)

print("Plots generated successfully!")


#TYLER USE THIS: --start_date 2011-01-02 --end_date 2011-11-20 --variable t --ensemble_type perturbed

