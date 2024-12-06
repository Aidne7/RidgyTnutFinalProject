import argparse
import os
import pickle
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import false_discovery_control
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# Parse command-line arguments for configuring the analysis
parser = argparse.ArgumentParser(description="Examine Normality Test P-Values with FDR Control")
parser.add_argument("--start_date", type=str, required=True, help="Start date in YYYY-MM-DD format")
parser.add_argument("--end_date", type=str, required=True, help="End date in YYYY-MM-DD format")
parser.add_argument("--variable", type=str, required=True, help="Variable name to analyze")
parser.add_argument("--ensemble_type", type=str, choices=["reference", "perturbed"], required=True, help="Type of ensemble")
parser.add_argument("--interval", type=int, default=1, help="Interval (in days) between data files to process")
args = parser.parse_args()

# Generate a list of dates based on the provided start date, end date, and interval
start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
dates = [start_date + timedelta(days=i) for i in range(0, (end_date - start_date).days + 1, args.interval)]

# Load p-value data from pickle files for the specified dates
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

# Stack the p-value arrays into a 4D array (time, levels, latitude, longitude)
pvals_4d = np.stack(pval_arrays)

# Flatten the 4D p-value array into 1D for False Discovery Rate (FDR) correction
pvals_flat = pvals_4d.ravel()

# Perform Benjamini-Yekutieli FDR correction on the p-values
adjusted_pvals = false_discovery_control(pvals_flat, method="by")

# Identify rejections of the null hypothesis based on a significance threshold (alpha)
alpha = 0.05
rejected = adjusted_pvals < alpha

# Reshape the adjusted p-values and rejection results back into 4D
adjusted_pvals_4d = adjusted_pvals.reshape(pvals_4d.shape)
rejected_4d = rejected.reshape(pvals_4d.shape)

# Print the dimensions of the adjusted arrays for verification
print("Adjusted p-values shape:", adjusted_pvals_4d.shape)
print("Rejected null hypothesis shape:", rejected_4d.shape)

def create_plot(data, title, xlabel, ylabel, filename, x_extent, y_extent, y_ticks=None):
    """
    Generate and save a heatmap plot for the given data.
    """
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
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=10))  # Show major ticks every 10 days
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=5))

    plt.xticks(rotation=45)

    # Add title, labels, and optional y-ticks
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    if y_ticks:
        ax.set_yticks(y_ticks)

    # Add a colorbar with a label
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Percentage of Null Hypothesis Rejections", fontsize=12)

    # Add gridlines for clarity
    ax.grid(color='white', linestyle='--', linewidth=0.5)

    # Save the plot
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


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
    y_extent=[0, 1],
)

# (ii) Null hypothesis rejections by latitude over time
rejections_time_lat = np.sum(rejected_4d, axis=(1, 2))
percent_rejections_time_lat = (rejections_time_lat / (rejected_4d.shape[2] * rejected_4d.shape[3])) * 100

create_plot(
    percent_rejections_time_lat,
    f"Rejections by Time and Latitude for {args.variable} ({args.ensemble_type} ensemble)",
    "Time",
    "Latitude Index",
    f"{args.variable}_{args.ensemble_type}_time_lat.png",
    x_extent=[mdates.date2num(time_steps[0]), mdates.date2num(time_steps[-1])],
    y_extent=[-96, rejected_4d.shape[3]],
)

# (iii) Null hypothesis rejections by model levels (pressure) over time
rejections_time_pressure = np.sum(rejected_4d, axis=(2, 3))
percent_rejections_time_pressure = (rejections_time_pressure / (rejected_4d.shape[2] * rejected_4d.shape[3])) * 100

create_plot(
    percent_rejections_time_pressure,
    f"Rejections by Time and Theoretical Pressure for {args.variable} ({args.ensemble_type} ensemble)",
    "Time",
    "Model Levels (Theoretical Pressure)",
    f"{args.variable}_{args.ensemble_type}_time_pressure.png",
    x_extent=[mdates.date2num(time_steps[0]), mdates.date2num(time_steps[-1])], #set y extent
    y_extent=[0, rejected_4d.shape[1]],
)

print("Plots generated successfully!")
