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