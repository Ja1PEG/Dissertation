import re
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_game_data(file_pattern):
    files = glob.glob(file_pattern)
    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {file_pattern}")

    all_average_times = []
    num_snakes_list = []
    num_ladders_list = []

    for filename in files:
        match = re.search(r"snakes_(\d+)_ladders_(\d+)_snake_len_(\d+)_ladder_len_(\d+)", filename)
        if not match:
            print(f"Skipping file with invalid format: {filename}")
            continue

        num_snakes, num_ladders, snake_length, ladder_length = map(int, match.groups())

        try:
            df = pd.read_csv(filename)
        except Exception as e:
            print(f"Error reading file {filename}: {e}")
            continue

        # Calculate average game time for each board
        average_game_times = df.mean(axis=0)
        all_average_times.append(average_game_times.values)  # Store as NumPy array
        num_snakes_list.append(num_snakes)
        num_ladders_list.append(num_ladders)

    # 1. Boxplots grouped by N_S and N_L
    data = []
    labels = []
    for i, times in enumerate(all_average_times):
        data.append(times)
        labels.append(f'Snakes: {num_snakes_list[i]}, Ladders: {num_ladders_list[i]}')

    plt.figure(figsize=(12, 8))
    plt.boxplot(data, labels=labels)
    plt.title('Distribution of Average Game Times Grouped by N_S and N_L')
    plt.xlabel('Snake and Ladder Configuration')
    plt.ylabel('Average Game Time')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 2. Heatmap of Average Game Times
    unique_num_snakes = sorted(set(num_snakes_list))
    unique_num_ladders = sorted(set(num_ladders_list))
    heatmap_data = np.zeros((len(unique_num_snakes), len(unique_num_ladders)))

    for i, n_snakes in enumerate(unique_num_snakes):
        for j, n_ladders in enumerate(unique_num_ladders):
            indices = [k for k, (s, l) in enumerate(zip(num_snakes_list, num_ladders_list)) if s == n_snakes and l == n_ladders]
            heatmap_data[i, j] = np.mean([all_average_times[k] for k in indices])

    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap_data, cmap='viridis', interpolation='nearest')
    plt.title('Heatmap of Average Game Times')
    plt.xlabel('Number of Ladders (N_L)')
    plt.ylabel('Number of Snakes (N_S)')
    plt.xticks(range(len(unique_num_ladders)), unique_num_ladders)
    plt.yticks(range(len(unique_num_snakes)), unique_num_snakes)
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    # 3. Trend lines plot
    plt.figure(figsize=(12, 8))
    for i, times in enumerate(all_average_times):
        plt.plot(range(1, len(times) + 1), times, marker='o', label=f"N_S: {num_snakes_list[i]}, N_L: {num_ladders_list[i]}")

    plt.title("Trend in Average Game Times for Different Configurations")
    plt.xlabel("Board Number")
    plt.ylabel("Average Game Time")
    plt.xticks(range(1, len(times) + 1))
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    file_pattern = "snakes_*_ladders_*_snake_len_*_ladder_len_*.csv"
    analyze_game_data(file_pattern)