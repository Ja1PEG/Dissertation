import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BOARD_SIZE = 100

# Helper function for length sampling
def sample_length(start_pos, max_length, distribution, is_snake=True):
    if distribution == "uniform":
        return random.randint(1, max_length)
    elif distribution == "normal":
        mean = max_length / 2
        std_dev = max_length / 6
        length = int(round(np.random.normal(mean, std_dev)))
        return max(1, min(length, max_length))
    elif distribution == "exponential":
        scale = max_length / 3
        length = int(round(np.random.exponential(scale)))
        return max(1, min(length, max_length))
    else:
        raise ValueError("Invalid distribution type")

# Generate a board
def generate_board(num_snakes, num_ladders, distribution):
    snakes_and_ladders = {}
    while len(snakes_and_ladders) < (num_snakes + num_ladders):
        is_snake = len(snakes_and_ladders) < num_snakes
        start_pos = random.randint(2, BOARD_SIZE - 1)  # Exclude start and end tiles
        max_length = start_pos - 1 if is_snake else BOARD_SIZE - start_pos
        if max_length <= 0:
            continue

        length = sample_length(start_pos, max_length, distribution, is_snake)
        end_pos = start_pos - length if is_snake else start_pos + length
        if end_pos < 1 or end_pos > BOARD_SIZE:
            continue

        if start_pos in snakes_and_ladders or end_pos in snakes_and_ladders.values():
            if random.random() < 0.5:
                del_key = random.choice(list(snakes_and_ladders.keys()))
                del snakes_and_ladders[del_key]
            continue

        snakes_and_ladders[start_pos] = end_pos
    return snakes_and_ladders

# Simulate a single game
def play_game(snakes_and_ladders):
    position = 0
    moves = 0
    while position < BOARD_SIZE:
        dice_roll = random.randint(1, 6)
        position += dice_roll
        moves += 1
        if position in snakes_and_ladders:
            position = snakes_and_ladders[position]
        if position > BOARD_SIZE:
            position = BOARD_SIZE
    return moves

# Run simulations for a board
def simulate_games(snakes_and_ladders, num_simulations):
    game_times = [play_game(snakes_and_ladders) for _ in range(num_simulations)]
    return game_times

# Run simulations and generate CSVs
def run_simulation(num_boards, num_simulations, num_snakes, num_ladders, distributions):
    all_results = {}
    for distribution in distributions:
        boards = {}
        game_times_all_boards = []
        
        # Generate boards and simulate games
        for board_index in range(num_boards):
            snakes_and_ladders = generate_board(num_snakes, num_ladders, distribution)
            boards[f"Board{board_index + 1}"] = [
                f"{start}->{end}" for start, end in snakes_and_ladders.items()
            ]
            game_times = simulate_games(snakes_and_ladders, num_simulations)
            game_times_all_boards.append(game_times)

        # Save board layouts to a CSV
        board_df = pd.DataFrame.from_dict(boards, orient="columns")
        board_df.to_csv(f"{distribution}.csv", index=False)

        # Store simulation results
        all_results[distribution] = game_times_all_boards

    return all_results

# Plotting functions
def plot_board_averages(all_results):
    """
    Plots the average game time for each board (X-axis: Board Numbers, Y-axis: Average Game Time)
    for each sampling distribution.
    """
    for distribution, game_times_all_boards in all_results.items():
        # Calculate the average game time for each board
        board_averages = [np.mean(game_times) for game_times in game_times_all_boards]
        board_numbers = [f"Board {i + 1}" for i in range(len(board_averages))]

        # Plot the board averages
        plt.figure(figsize=(10, 6))
        plt.bar(board_numbers, board_averages, color="skyblue", edgecolor="black", alpha=0.8)
        
        # Annotate the bars with the average values
        for i, avg in enumerate(board_averages):
            plt.text(i, avg + 0.5, f"{avg:.2f}", ha="center", fontsize=9)

        # Finalise the plot
        plt.title(f"Average Game Time for Each Board ({distribution})")
        plt.xlabel("Board Number")
        plt.ylabel("Average Game Time (Moves)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"board_averages_{distribution}.png")
        plt.close()

def plot_comparative_aggregate_averages(all_results):
    """
    Plots the aggregate average game time for each sampling distribution.
    X-axis: Sampling Distributions
    Y-axis: Aggregate Average Game Time
    """
    # Calculate the overall average game time for each distribution
    aggregate_averages = {
        distribution: np.mean([np.mean(game_times) for game_times in game_times_all_boards])
        for distribution, game_times_all_boards in all_results.items()
    }

    # Create the bar plot
    plt.figure(figsize=(8, 6))
    plt.bar(
        aggregate_averages.keys(),
        aggregate_averages.values(),
        color=["skyblue", "salmon", "limegreen"],
        edgecolor="black",
        alpha=0.8
    )

    # Annotate the bars with average values
    for i, avg in enumerate(aggregate_averages.values()):
        plt.text(i, avg + 0.5, f"{avg:.2f}", ha="center", fontsize=10)

    # Finalise the plot
    plt.title("Aggregate Average Game Time Across Sampling Distributions")
    plt.xlabel("Sampling Distribution")
    plt.ylabel("Aggregate Average Game Time (Moves)")
    plt.tight_layout()
    plt.savefig("comparative_aggregate_average_game_times.png")
    plt.close()

def calculate_relative_metrics(snakes_and_ladders):
    """
    Calculate Avg(LengthSnake - LengthLadder) and Avg(LengthSnake / LengthLadder)
    for a given board.
    """
    snakes = [start - end for start, end in snakes_and_ladders.items() if start > end]
    ladders = [end - start for start, end in snakes_and_ladders.items() if start < end]

    if not snakes or not ladders:
        return None, None

    avg_difference = np.mean(snakes) - np.mean(ladders)
    avg_ratio = np.mean(snakes) / np.mean(ladders) if np.mean(ladders) != 0 else None
    return avg_difference, avg_ratio

def plot_relative_logic(all_results, num_boards, num_snakes, num_ladders, distributions):
    """
    Plots the relationship between game time and Avg(LengthSnake - LengthLadder)
    and Avg(LengthSnake / LengthLadder) for all sampling distributions.
    """
    avg_game_times = []
    avg_differences = []
    avg_ratios = []
    labels = []

    for distribution, game_times_all_boards in all_results.items():
        for board_index in range(num_boards):
            # Recreate the board to calculate metrics
            snakes_and_ladders = generate_board(num_snakes, num_ladders, distribution)
            avg_difference, avg_ratio = calculate_relative_metrics(snakes_and_ladders)

            if avg_difference is not None and avg_ratio is not None:
                avg_game_time = np.mean(game_times_all_boards[board_index])
                avg_game_times.append(avg_game_time)
                avg_differences.append(avg_difference)
                avg_ratios.append(avg_ratio)
                labels.append(distribution.capitalize())

    # Plot Avg(LengthSnake - LengthLadder) vs Game Time
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=avg_differences, y=avg_game_times, hue=labels, palette="Set2")
    plt.title("Game Time vs Avg(LengthSnake - LengthLadder)")
    plt.xlabel("Avg(LengthSnake - LengthLadder)")
    plt.ylabel("Average Game Time (Moves)")
    plt.legend(title="Sampling Distribution")
    plt.tight_layout()
    plt.savefig("game_time_vs_length_difference.png")
    plt.close()

    # Plot Avg(LengthSnake / LengthLadder) vs Game Time
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=avg_ratios, y=avg_game_times, hue=labels, palette="Set2")
    plt.title("Game Time vs Avg(LengthSnake / LengthLadder)")
    plt.xlabel("Avg(LengthSnake / LengthLadder)")
    plt.ylabel("Average Game Time (Moves)")
    plt.legend(title="Sampling Distribution")
    plt.tight_layout()
    plt.savefig("game_time_vs_length_ratio.png")
    plt.close()
    
def plot_relative_logic_lineplots(all_results, num_boards, num_snakes, num_ladders, distributions):
    """
    Creates line plots showing trends in game times as functions of relative metrics.
    """
    avg_game_times = []
    avg_differences = []
    avg_ratios = []
    labels = []

    for distribution, game_times_all_boards in all_results.items():
        for board_index in range(num_boards):
            # Recreate the board to calculate metrics
            snakes_and_ladders = generate_board(num_snakes, num_ladders, distribution)
            avg_difference, avg_ratio = calculate_relative_metrics(snakes_and_ladders)

            if avg_difference is not None and avg_ratio is not None:
                avg_game_time = np.mean(game_times_all_boards[board_index])
                avg_game_times.append(avg_game_time)
                avg_differences.append(avg_difference)
                avg_ratios.append(avg_ratio)
                labels.append(distribution.capitalize())

    # Line plot for Avg(LengthSnake - LengthLadder)
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=avg_differences, y=avg_game_times, hue=labels, marker="o", palette="Set2")
    plt.title("Game Time vs Avg(LengthSnake - LengthLadder)")
    plt.xlabel("Avg(LengthSnake - LengthLadder)")
    plt.ylabel("Average Game Time")
    plt.legend(title="Sampling Distribution")
    plt.tight_layout()
    plt.savefig("game_time_vs_length_difference_lineplot.png")
    plt.close()

    # Line plot for Avg(LengthSnake / LengthLadder)
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=avg_ratios, y=avg_game_times, hue=labels, marker="o", palette="Set2")
    plt.title("Game Time vs Avg(LengthSnake / LengthLadder)")
    plt.xlabel("Avg(LengthSnake / LengthLadder)")
    plt.ylabel("Average Game Time")
    plt.legend(title="Sampling Distribution")
    plt.tight_layout()
    plt.savefig("game_time_vs_length_ratio_lineplot.png")
    plt.close()

def log_final_results(all_results, num_boards, num_snakes, num_ladders, distributions):
    """
    Logs the final results into a CSV file summarising key metrics.
    """
    final_results = []

    for distribution, game_times_all_boards in all_results.items():
        for board_index in range(num_boards):
            # Recreate the board to calculate metrics
            snakes_and_ladders = generate_board(num_snakes, num_ladders, distribution)
            avg_difference, avg_ratio = calculate_relative_metrics(snakes_and_ladders)

            if avg_difference is not None and avg_ratio is not None:
                avg_game_time = np.mean(game_times_all_boards[board_index])
                final_results.append({
                    "Distribution": distribution.capitalize(),
                    "Board Number": board_index + 1,
                    "Average Game Time": avg_game_time,
                    "Avg(LengthSnake - LengthLadder)": avg_difference,
                    "Avg(LengthSnake / LengthLadder)": avg_ratio
                })

    # Convert to DataFrame and save
    results_df = pd.DataFrame(final_results)
    results_df.to_csv("final_results.csv", index=False)

def log_board_details(all_results, num_boards, num_snakes, num_ladders, distributions):
    """
    Logs detailed snake/ladder configurations for each board into CSV files.
    """
    for distribution in distributions:
        board_details = []

        for board_index in range(num_boards):
            # Recreate the board to calculate metrics
            snakes_and_ladders = generate_board(num_snakes, num_ladders, distribution)
            avg_difference, avg_ratio = calculate_relative_metrics(snakes_and_ladders)

            # Record snake/ladder positions and metrics
            board_data = {
                "Board Number": board_index + 1,
                "Snakes": "; ".join([f"{start}->{end}" for start, end in snakes_and_ladders.items() if start > end]),
                "Ladders": "; ".join([f"{start}->{end}" for start, end in snakes_and_ladders.items() if start < end]),
                "Avg(LengthSnake - LengthLadder)": avg_difference,
                "Avg(LengthSnake / LengthLadder)": avg_ratio
            }
            board_details.append(board_data)

        # Save board details to a CSV
        details_df = pd.DataFrame(board_details)
        details_df.to_csv(f"{distribution}_board_details.csv", index=False)


if __name__ == "__main__":
    num_boards = 10
    num_simulations = 1000
    num_snakes = 10
    num_ladders = 10
    distributions = ["uniform", "normal", "exponential"]

     # Run simulations and save results
    all_results = run_simulation(num_boards, num_simulations, num_snakes, num_ladders, distributions)

    # Log final results
    log_final_results(all_results, num_boards, num_snakes, num_ladders, distributions)

    # Log per-board details
    log_board_details(all_results, num_boards, num_snakes, num_ladders, distributions)

    # Plot average game time for each board
    plot_board_averages(all_results)

    # Plot aggregate average game times
    plot_comparative_aggregate_averages(all_results)

    # Plot relative logic metrics
    plot_relative_logic(all_results, num_boards, num_snakes, num_ladders, distributions)
    plot_relative_logic_lineplots(all_results, num_boards, num_snakes, num_ladders, distributions)

    

