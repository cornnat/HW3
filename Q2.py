import os
import numpy as np
import matplotlib.pyplot as plt

######################################
# Part A
###########################################

# Define paths
base_dir = "/root/Desktop/host/HW3/Local_density_of_states_near_band_edge"
output_dir = os.path.join(base_dir, "local_density_of_states_heatmap")
os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

# Process each text file
for i in range(11):  # Numbers 0 to 10
    file_name = f"local_density_of_states_for_level_{i}.txt"
    file_path = os.path.join(base_dir, file_name)

    if not os.path.exists(file_path):
        print(f"File {file_path} not found, skipping.")
        continue

    # Read and clean data (handle potential trailing commas)
    with open(file_path, "r") as f:
        lines = [line.strip().rstrip(",") for line in f]  # Strip spaces & remove trailing commas
        clean_data = [list(map(float, line.split(","))) for line in lines]  # Convert to float

    data = np.array(clean_data)  # Convert to NumPy array

    # Generate heatmap
    plt.figure(figsize=(6, 5))
    plt.imshow(data, cmap="inferno", aspect="auto")  # 'inferno' for better contrast
    plt.colorbar(label="Electron Density")
    plt.title(f"Local Electron Density (Level {i})")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")

    # Save the figure
    output_path = os.path.join(output_dir, f"heatmap_level_{i}.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")

print("All heatmaps generated.")








########################################################
# part B
##########################################################

from mpl_toolkits.mplot3d import Axes3D

# Define output directory for surface plots
height_output_dir = os.path.join(base_dir, "local_density_of_states_height")
os.makedirs(height_output_dir, exist_ok=True)  # Create output directory if it doesn't exist

# Generate surface plots for each file
for i in range(11):  # Numbers 0 to 10
    file_name = f"local_density_of_states_for_level_{i}.txt"
    file_path = os.path.join(base_dir, file_name)

    if not os.path.exists(file_path):
        print(f"File {file_path} not found, skipping.")
        continue

    # Read and clean data (reuse from heatmap code)
    with open(file_path, "r") as f:
        lines = [line.strip().rstrip(",") for line in f]  # Strip spaces & remove trailing commas
        clean_data = [list(map(float, line.split(","))) for line in lines]  # Convert to float

    data = np.array(clean_data)  # Convert to NumPy array

    # Create grid for surface plot
    x = np.arange(data.shape[1])  # X-axis indices
    y = np.arange(data.shape[0])  # Y-axis indices
    X, Y = np.meshgrid(x, y)  # Create coordinate grid

    # Generate surface plot
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, data, cmap="inferno", edgecolor='none')  # Create surface plot

    # Labels and title
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Electron Density")
    ax.set_title(f"Local Electron Density (Level {i})")

    # Save the figure
    output_path = os.path.join(height_output_dir, f"height_profile_level_{i}.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")

print("All surface plots generated.")







########################################################
# Part C
########################################################

# Define the subregion (adjust these values as needed)
subregion_x_start, subregion_x_end = 5, 15  # X-range of subregion
subregion_y_start, subregion_y_end = 5, 15  # Y-range of subregion

# Store the average LDOS for each file
indices = []
average_ldos_values = []

# Process each text file again
for i in range(11):  # Numbers 0 to 10
    file_name = f"local_density_of_states_for_level_{i}.txt"
    file_path = os.path.join(base_dir, file_name)

    if not os.path.exists(file_path):
        print(f"File {file_path} not found, skipping.")
        continue

    # Read and clean data
    with open(file_path, "r") as f:
        lines = [line.strip().rstrip(",") for line in f]  # Strip spaces & remove trailing commas
        clean_data = [list(map(float, line.split(","))) for line in lines]  # Convert to float

    data = np.array(clean_data)  # Convert to NumPy array

    # Extract subregion and compute average LDOS
    subregion = data[subregion_y_start:subregion_y_end, subregion_x_start:subregion_x_end]
    avg_ldos = np.mean(subregion)

    # Store results
    indices.append(i)
    average_ldos_values.append(avg_ldos)

    print(f"Level {i}: Average LDOS in subregion = {avg_ldos:.4f}")

# Plot the average LDOS changes across all levels
plt.figure(figsize=(7, 5))
plt.plot(indices, average_ldos_values, marker='o', linestyle='-', color='b', label="Avg LDOS in Subregion")
plt.xlabel("Level Index")
plt.ylabel("Average Local Density of States")
plt.title("Variation of Average LDOS in Selected Subregion")
plt.legend()
plt.grid(True)

# Save plot
analysis_output_dir = os.path.join(base_dir, "ldos_subregion_analysis")
os.makedirs(analysis_output_dir, exist_ok=True)
analysis_plot_path = os.path.join(analysis_output_dir, "average_ldos_trend.png")
plt.savefig(analysis_plot_path, dpi=300)
plt.close()

print(f"Saved subregion analysis plot: {analysis_plot_path}")
print("Subregion analysis completed.")

# This code is performing a local subregion analysis on the local density of states (LDOS) across different energy levels (0–10). Here’s a breakdown:
# 1. it defines a Subregion:
#       - it selects a rectangular region within the LDOS matrix (from x=5:15, y=5:15).
#       - this subregion represents a localized portion of the material where the LDOS is analyzed.
#       - then it computes the avg LDOS in the subregion:
# 
# 2. it extracts LDOS values from the selected subregion.
#       - computes the mean LDOS in this subregion.
#       - this represents the local electronic state availability in this area.
#       - tracks LDOS Changes Across Levels (0–10):
# 
# 3. then the average LDOS is computed for each file (corresponding to different levels).
#         and the trend is plotted to show how the LDOS varies across these levels.
#       - this shows us if the local electronic density is increasing, decreasing, or fluctuating.

# if the average LDOS in the subregion increases with level index:
#       - more electronic states are available in this region at higher levels.
#       - this suggests localization of electrons, possibly due to trapping effects or 
#         impurity states.
# 
# if the LDOS decreases with level index:
#       - fewer states are present, implying that electrons are becoming delocalized.
#       - this could indicate energy band widening, leading to easier conduction.
# 
# 
# if LDOS oscillates across levels:
#       - this could indicate quantum interference effects or fluctuating states due to disorder in the material.