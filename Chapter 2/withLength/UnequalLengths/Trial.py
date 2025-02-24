import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BOARD_SIZE = 100

# Generate a board with fixed lengths for snakes and ladders
def generate_board_fixed_lengths(num_snakes, num_ladders, snake_length, ladder_length):
    """
    Generates a board with fixed lengths for all snakes and ladders.
    """
    snakes_and_ladders = {}
    snake_positions = []
    ladder_positions = []

    # Assign snakes
    for _ in range(num_snakes):
        while True:
            start = random.randint(snake_length, BOARD_SIZE - 1)
            end = start - snake_length
            if start not in snakes_and_ladders and end not in snakes_and_ladders.values():
                snakes_and_ladders[start] = end
                snake_positions.append((start, end))
                break

    # Assign ladders
    for _ in range(num_ladders):
        while True:
            start = random.randint(2, BOARD_SIZE - ladder_length)
            end = start + ladder_length
            if start not in snakes_and_ladders and end not in snakes_and_ladders.values():
                snakes_and_ladders[start] = end
                ladder_positions.append((start, end))
                break

    return snakes_and_ladders, snake_positions, ladder_positions

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

# Simulate games for pairs of lengths, log data, and generate plots
def simulate_fixed_length_pairs(num_boards, num_simulations, num_snakes, num_ladders, length_pairs):
    all_results = []
    all_game_times = []  # To capture all game times for frequency distributions
    board_details = []

    for pair in length_pairs:
        snake_length, ladder_length = pair
        results = []
        game_times_for_pair = []  # To capture all game times for this pair

        for board_num in range(num_boards):
            snakes_and_ladders, snake_positions, ladder_positions = generate_board_fixed_lengths(
                num_snakes, num_ladders, snake_length, ladder_length
            )
            game_times = [play_game(snakes_and_ladders) for _ in range(num_simulations)]
            avg_time = np.mean(game_times)
            results.append(avg_time)
            game_times_for_pair.extend(game_times)  # Collect all game times for frequency distributions

            # Log board details
            board_details.append({
                "Board Number": f"Board {board_num + 1}",
                "Snakes": "; ".join([f"{s[0]}->{s[1]}" for s in snake_positions]),
                "Ladders": "; ".join([f"{l[0]}->{l[1]}" for l in ladder_positions]),
                "Average Game Time": avg_time
            })

        # Store the average game times for plotting
        all_results.append({
            "Pair": f"Snake{snake_length}_Ladder{ladder_length}",
            "Average Times": results
        })

        # Log all game times for frequency distribution later
        all_game_times.append({
            "Pair": f"Snake{snake_length}_Ladder{ladder_length}",
            "Game Times": game_times_for_pair
        })

        # Save board details to CSV
        board_df = pd.DataFrame(board_details)
        board_df.to_csv(f"approach_1_board_details_Snake{snake_length}_Ladder{ladder_length}.csv", index=False)

    # Plot the frequency distributions for all pairs
    for result in all_game_times:
        plt.figure(figsize=(10, 6))
        sns.histplot(result["Game Times"], bins=30, kde=True, color="blue", edgecolor="black")
        plt.title(f"Game Time Distribution for {result['Pair']} (Frequency Distribution)")
        plt.xlabel("Game Time (Moves)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(f"game_time_distribution_{result['Pair']}.png")

    # Plot results for average game times (Bar Plot) with values on bars
    plt.figure(figsize=(12, 8))
    for result in all_results:
        avg_time = np.mean(result["Average Times"])
        plt.bar(result["Pair"], avg_time, alpha=0.6)
        plt.text(result["Pair"], avg_time, f'{avg_time:.2f}', ha='center', va='bottom')

    plt.title("Average Game Times for Fixed Length Pairs (Bar Plot)")
    plt.xlabel("Length Pairs (Snake, Ladder)")
    plt.ylabel("Average Game Time (Moves)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("approach_1_fixed_length_pairs_barplot.png")

    # Prepare data for heatmap (Game Times per Pair and Board)
    heatmap_data = pd.DataFrame({
        result["Pair"]: result["Average Times"]
        for result in all_results
    }, index=[f"Board {i+1}" for i in range(num_boards)])

    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlGnBu", cbar=True)
    plt.title("Heatmap of Average Game Times for Fixed Length Pairs")
    plt.xlabel("Length Pairs (Snake, Ladder)")
    plt.ylabel("Board Number")
    plt.tight_layout()
    plt.savefig("approach_1_fixed_length_pairs_heatmap.png")

    # Prepare data for box plot (Game Times per Pair)
    plot_data = []
    plot_labels = []

    for result in all_game_times:
        plot_data.extend(result["Game Times"])
        plot_labels.extend([result["Pair"]] * num_simulations * num_boards)

    # Plot box plot
    plt.figure(figsize=(12, 8))
    sns.boxplot(x=plot_labels, y=plot_data, palette="Set2")
    plt.title("Game Time Distributions by Fixed Length Pairs (Box Plot)")
    plt.xlabel("Length Pairs (Snake, Ladder)")
    plt.ylabel("Game Time (Moves)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("approach_1_fixed_length_pairs_boxplot.png")

if __name__ == "__main__":
    num_boards = 20
    num_simulations = 1000
    num_snakes = 5
    num_ladders = 5

    # Define pairs of lengths (Snake Length, Ladder Length)
    length_pairs = [(5, 10), (10, 20), (20, 40), (40, 20), (20, 10), (10, 5)]

    simulate_fixed_length_pairs(num_boards, num_simulations, num_snakes, num_ladders, length_pairs)
    