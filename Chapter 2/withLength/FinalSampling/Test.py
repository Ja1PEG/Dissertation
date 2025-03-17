import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BOARD_SIZE = 100

# Helper function for length sampling with fixed max_length
def sample_length_with_fixed_max(start_pos, max_length, distribution, is_snake=True):
    """
    Samples the length of a snake or ladder using a distribution,
    ensuring that the resulting position is valid.
    """
    while True:
        if distribution == "uniform":
            length = random.randint(1, max_length)
        elif distribution == "normal":
            mean = max_length / 2
            std_dev = max_length / 6
            length = int(round(np.random.normal(mean, std_dev)))
        elif distribution == "exponential":
            scale = max_length / 3
            length = int(round(np.random.exponential(scale)))
        else:
            raise ValueError("Invalid distribution type")

        # Ensure length is within valid bounds
        if length < 1 or length > max_length:
            continue

        end_pos = start_pos - length if is_snake else start_pos + length

        # Check if the end position is valid
        if 1 <= end_pos <= BOARD_SIZE:
            return length

# Generate a board using sampling
def generate_board_with_sampling(num_snakes, num_ladders, max_length, distribution):
    """
    Generates a board using the sampling functions, ensuring no overlaps
    and valid lengths for snakes and ladders.
    """
    snakes_and_ladders = {}

    while len(snakes_and_ladders) < (num_snakes + num_ladders):
        is_snake = len(snakes_and_ladders) < num_snakes
        start_pos = random.randint(2, BOARD_SIZE - 1)  # Exclude start and end tiles

        # Sample length
        length = sample_length_with_fixed_max(
            start_pos, max_length, distribution, is_snake
        )
        end_pos = start_pos - length if is_snake else start_pos + length

        # Check for overlap
        if start_pos in snakes_and_ladders or end_pos in snakes_and_ladders.values():
            continue

        # Add to board
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


# Log final results into a summary CSV
def log_final_results(all_results, num_boards, num_snakes, num_ladders, distributions):
    """
    Logs the final results into a CSV file summarising key metrics.
    """
    final_results = []

    for distribution, game_times_all_boards in all_results.items():
        for board_index in range(num_boards):
            # Calculate average Game Duration
            avg_game_time = np.mean(game_times_all_boards[board_index])
            final_results.append(
                {
                    "Distribution": distribution.capitalize(),
                    "Board Number": board_index + 1,
                    "Average Game Duration": avg_game_time,
                }
            )

    # Convert to DataFrame and save
    results_df = pd.DataFrame(final_results)
    results_df.to_csv("final_results.csv", index=False)

# Log board-specific details into individual CSV files
def log_board_details(all_results, num_boards, num_snakes, num_ladders, distributions):
    """
    Logs detailed snake/ladder configurations for each board into CSV files.
    """
    for distribution in distributions:
        board_details = []

        for board_index in range(num_boards):
            # Generate the board to calculate metrics
            snakes_and_ladders = generate_board_with_sampling(
                num_snakes, num_ladders, max_length, distribution
            )

            # Record snake/ladder positions
            board_data = {
                "Board Number": board_index + 1,
                "Snakes": "; ".join(
                    [
                        f"{start}->{end}"
                        for start, end in snakes_and_ladders.items()
                        if start > end
                    ]
                ),
                "Ladders": "; ".join(
                    [
                        f"{start}->{end}"
                        for start, end in snakes_and_ladders.items()
                        if start < end
                    ]
                ),
            }
            board_details.append(board_data)

        # Save board details to a CSV
        details_df = pd.DataFrame(board_details)
        details_df.to_csv(f"{distribution}_board_details.csv", index=False)

# Plotting functions
def plot_board_averages(all_results):
    """
    Plots the average Game Duration for each board (X-axis: Board Numbers, Y-axis: Average Game Duration)
    for each sampling distribution.
    """
    for distribution, game_times_all_boards in all_results.items():
        # Calculate the average Game Duration for each board
        board_averages = [np.mean(game_times) for game_times in game_times_all_boards]
        board_numbers = [f"Board {i + 1}" for i in range(len(board_averages))]

        # Plot the board averages
        plt.figure(figsize=(10, 6))
        plt.bar(
            board_numbers, board_averages, color="skyblue", edgecolor="black", alpha=0.8
        )

        # Annotate the bars with the average values
        for i, avg in enumerate(board_averages):
            plt.text(i, avg + 0.5, f"{avg:.2f}", ha="center", fontsize=9)

        # Finalise the plot
        plt.title(f"Average Game Duration for Each Board ({distribution})")
        plt.xlabel("Board Number")
        plt.ylabel("Average Game Duration (Moves)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"board_averages_{distribution}.png")
        plt.close()
        
# Plot full distribution of Game Durations for multiple layouts (separated plots with aggregated overlay)
def plot_multiple_distributions(all_results, num_boards_to_plot):
    """
    Plots the distribution of Game Durations for multiple board layouts,
    with a separate plot for each sampling distribution.
    Each plot also includes an overlay of the aggregated distribution.
    """
    colors = ["blue", "green", "red"]
    distribution_names = ["Uniform", "Normal", "Exponential"]

    for i, distribution in enumerate(all_results.keys()):
        plt.figure(figsize=(12, 6))

        # --- Individual Board Layouts ---
        for board_index in range(num_boards_to_plot):
            game_times = all_results[distribution][board_index]
            sns.kdeplot(
                game_times,
                label=f"Board {board_index + 1}",
                color=colors[i],
                alpha=0.5,
                linestyle="--",  # Use dashed lines for individual boards
            )

        # --- Aggregated Distribution ---
        all_game_times = [
            game_time
            for game_times_one_board in all_results[distribution]
            for game_time in game_times_one_board
        ]
        sns.kdeplot(
            all_game_times,
            label=f"{distribution_names[i]} (Aggregated)",
            color=colors[i],
            linewidth=2,
        )

        plt.title(f"Game Duration Distributions for Multiple Layouts ({distribution_names[i]})")
        plt.xlabel("Game Duration (Moves)")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"multiple_distributions_{distribution}.png")
        plt.close()

# Plot aggregated distributions for all board layouts (overlayed)
def plot_aggregated_distributions_overlay(all_results):
    """
    Plots the aggregated distribution of Game Durations for all board layouts,
    overlaying the distributions for each sampling method on the same plot.
    """
    plt.figure(figsize=(10, 6))
    colors = {"uniform": "blue", "normal": "green", "exponential": "red"}
    distribution_names = {
        "uniform": "Uniform",
        "normal": "Normal",
        "exponential": "Exponential",
    }  # For clearer labels

    for distribution, game_times_all_boards in all_results.items():
        # Flatten the list of Game Durations from all boards into a single list
        all_game_times = [
            game_time
            for game_times_one_board in game_times_all_boards
            for game_time in game_times_one_board
        ]

        # Plot the aggregated distribution for the current sampling method
        sns.kdeplot(
            all_game_times,
            label=f"{distribution_names[distribution]} (Aggregated)",
            color=colors[distribution],
            alpha=0.7,
        )

    plt.title("Aggregated Game Duration Distributions (All Layouts)")
    plt.xlabel("Game Duration (Moves)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig("aggregated_distributions_overlayed.png")
    plt.close()

def plot_comparative_aggregate_averages(all_results):
    """
    Plots the aggregate average game Duration for each sampling distribution.
    X-axis: Sampling Distributions
    Y-axis: Aggregate Average Game Duration
    """
    # Calculate the overall average Game Duration for each distribution
    aggregate_averages = {
        distribution: np.mean(
            [np.mean(game_times) for game_times in game_times_all_boards]
        )
        for distribution, game_times_all_boards in all_results.items()
    }

    # Create the bar plot
    plt.figure(figsize=(8, 6))
    plt.bar(
        aggregate_averages.keys(),
        aggregate_averages.values(),
        color=["skyblue", "salmon", "limegreen"],
        edgecolor="black",
        alpha=0.8,
    )

    # Annotate the bars with average values
    for i, avg in enumerate(aggregate_averages.values()):
        plt.text(i, avg + 0.5, f"{avg:.2f}", ha="center", fontsize=10)

    # Finalise the plot
    plt.title("Aggregate Average Game Duration Across Sampling Distributions")
    plt.xlabel("Sampling Distribution")
    plt.ylabel("Aggregate Average Game Duration (Moves)")
    plt.tight_layout()
    plt.savefig("comparative_aggregate_average_game_times.png")

# Main program logic
if __name__ == "__main__":
    num_boards = 10
    num_simulations = 10000
    num_snakes = 10
    num_ladders = 10
    max_length = 40  # Fixed max_length
    distributions = ["uniform", "normal", "exponential"]

    # Store results for all boards
    all_results = {}

    for distribution in distributions:
        game_times_all_boards = []
        for board_index in range(num_boards):
            snakes_and_ladders = generate_board_with_sampling(
                num_snakes, num_ladders, max_length, distribution
            )
            game_times = simulate_games(snakes_and_ladders, num_simulations)
            game_times_all_boards.append(game_times)
        all_results[distribution] = game_times_all_boards

    # Plot distributions for multiple board layouts
    num_boards_to_plot = 10
    plot_multiple_distributions(all_results, num_boards_to_plot)

    # Log results
    log_final_results(all_results, num_boards, num_snakes, num_ladders, distributions)
    log_board_details(all_results, num_boards, num_snakes, num_ladders, distributions)

    # Additional plots and analyses can be called here
    plot_board_averages(all_results)
    plot_comparative_aggregate_averages(all_results)
    plot_aggregated_distributions_overlay(all_results)