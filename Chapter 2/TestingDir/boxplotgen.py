import glob
import os
import re

import matplotlib.pyplot as plt
import pandas as pd

def plot_boxplots_by_num_snakes(base_dir):
    """
    Reads data from CSV files in the given directory structure, groups them by the number of snakes,
    and generates box and whisker plots for each group in separate figures, saving them in respective folders.

    Args:
        base_dir (str): The base directory containing subdirectories named "snakes_{count}".
    """

    data_by_num_snakes = {}

    for snake_dir in os.listdir(base_dir):
        if not snake_dir.startswith("snakes_"):
            continue

        num_snakes = int(snake_dir.split("_")[1])
        snake_path = os.path.join(base_dir, snake_dir)

        for ladder_file in os.listdir(snake_path):
            if not ladder_file.startswith("ladders_") or not ladder_file.endswith(".csv"):
                continue

            num_ladders = int(ladder_file.split("_")[1].split(".")[0])

            try:
                file_path = os.path.join(snake_path, ladder_file)
                df = pd.read_csv(file_path)
                # Transpose the DataFrame to have boards as rows and simulations as columns
                df_transposed = df.transpose()

                # Calculate average game time for each board
                average_game_times = df_transposed.mean(axis=1)

                data_by_num_snakes.setdefault(num_snakes, []).append(average_game_times.values)
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                continue

    # Plotting
    for num_snakes, data in data_by_num_snakes.items():
        plt.figure(figsize=(8, 6))
        plt.boxplot(data)
        plt.title(f"Box and Whisker Plot of Average Game Times\nSnakes: {num_snakes}")
        plt.xlabel("Number of Ladders")
        plt.ylabel("Average Game Time")
        plt.xticks(range(1, len(data) + 1), range(5, 16))

        # Create directory for the plots if it doesn't exist
        plot_dir = os.path.join(base_dir, f"snakes_{num_snakes}")
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"snakes_{num_snakes}_boxplot.png"))
        plt.close()

if __name__ == "__main__":
    base_dir = "."  # Use the current directory as the base directory
    plot_boxplots_by_num_snakes(base_dir)