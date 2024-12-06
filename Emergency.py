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