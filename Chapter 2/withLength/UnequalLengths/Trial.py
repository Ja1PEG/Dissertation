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
    ladder_positions = set()  # Use a set to prevent duplicates

    # Assign snakes
    for _ in range(num_snakes):
        while True:
            start = random.randint(snake_length + 1, BOARD_SIZE - 1)
            end = start - snake_length
            if start not in snakes_and_ladders and end not in snakes_and_ladders.values():
                snakes_and_ladders[start] = end
                snake_positions.append((start, end))
                break

    # Assign ladders
    for _ in range(num_ladders):
        while True:
            start = random.randint(2, BOARD_SIZE - ladder_length - 1)
            end = start + ladder_length
            if start not in snakes_and_ladders and end not in snakes_and_ladders.values() and start not in ladder_positions:
                snakes_and_ladders[start] = end
                ladder_positions.add(start)
                break

    return snakes_and_ladders, snake_positions, list(ladder_positions)

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
    return moves

# Simulate games for pairs of lengths, log data, and generate plots
def simulate_fixed_length_pairs(num_boards, num_simulations, num_snakes, num_ladders, length_pairs):
    all_results = []
    all_game_times = []
    board_details = []

    for pair in length_pairs:
        snake_length, ladder_length = pair
        results = []
        game_times_for_pair = []

        for board_num in range(num_boards):
            snakes_and_ladders, snake_positions, ladder_positions = generate_board_fixed_lengths(
                num_snakes, num_ladders, snake_length, ladder_length
            )
            game_times = [play_game(snakes_and_ladders) for _ in range(num_simulations)]
            avg_time = np.mean(game_times)
            results.append(avg_time)
            game_times_for_pair.extend(game_times)

            # Log board details
            board_details.append({
                "Board Number": f"Board {board_num + 1}",
                "Snakes": "; ".join([f"{s[0]}->{s[1]}" for s in snake_positions]),
                "Ladders": "; ".join([f"{l}->{l + ladder_length}" for l in ladder_positions]),
                "Average Game Time": avg_time
            })

        # Store results for visualization and parsing
        all_results.append({
            "Pair": f"L$_{{s}}$={snake_length} L$_{{l}}$={ladder_length}",
            "Plain Pair": f"{snake_length}_{ladder_length}",
            "Average Times": results
        })

        # Store game times for frequency distribution
        all_game_times.append({
            "Pair": f"L$_{{s}}$={snake_length} L$_{{l}}$={ladder_length}",
            "Plain Pair": f"{snake_length}_{ladder_length}",
            "Game Times": game_times_for_pair
        })

        # Save board details to CSV
        pd.DataFrame(board_details).to_csv(
            f"approach_1_board_details_Snake{snake_length}_Ladder{ladder_length}.csv", index=False
        )

    # Plot frequency distributions
    for result in all_game_times:
        plt.figure(figsize=(10, 6))
        sns.histplot(result["Game Times"], bins=30, kde=True, color="blue", edgecolor="black")
        plt.title(f"Game Duration Distribution for {result['Pair']} (Frequency Distribution)")
        plt.xlabel("Game Duration (Turns)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(f"game_time_distribution_{result['Plain Pair']}.png")

    # Bar Plot: Average Game Times
    bar_data = []

    for result in all_results:
        avg_time = np.mean(result["Average Times"])
        snake_length, ladder_length = map(int, result["Plain Pair"].split("_"))
        diff_label = snake_length - ladder_length
        bar_data.append((diff_label, avg_time))

    # Sort bars
    bar_data.sort(key=lambda x: x[0])
    sorted_labels = [str(item[0]) for item in bar_data]
    sorted_avg_times = [item[1] for item in bar_data]

    # Plot sorted results
    plt.figure(figsize=(12, 8))
    bars = plt.bar(sorted_labels, sorted_avg_times, alpha=0.6)

    # Add values on bars
    for bar, avg_time in zip(bars, sorted_avg_times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{avg_time:.2f}', ha='center', va='bottom')

    plt.title("Average Game Duration for Fixed Length Pairs")
    plt.xlabel("Snake Length - Ladder Length")
    plt.ylabel("Average Game Time (Moves)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("approach_1_fixed_length_pairs_barplot.png")

    # Heatmap: Game Times per Pair and Board
    heatmap_data = pd.DataFrame({
        result["Pair"]: result["Average Times"]
        for result in all_results
    }, index=[f"Board {i+1}" for i in range(num_boards)])

    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlGnBu", cbar=True)
    plt.title("Heatmap of Average Game Times for Fixed Length Pairs")
    plt.xlabel("Length Pairs (Snake, Ladder)")
    plt.ylabel("Board Number")
    plt.tight_layout()
    plt.savefig("approach_1_fixed_length_pairs_heatmap.png")

    # Box Plot: Game Time Distributions
    plot_data = []
    plot_labels = []

    for result in all_game_times:
        plot_data.extend(result["Game Times"])
        plot_labels.extend([result["Pair"]] * num_simulations * num_boards)

    plt.figure(figsize=(12, 8))
    sns.boxplot(x=plot_labels, y=plot_data, palette="Set2")
    plt.title("Game Time Distributions by Fixed Length Pairs (Box Plot)")
    plt.xlabel("Length Pairs (Snake, Ladder)")
    plt.ylabel("Game Duration (Moves)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("approach_1_fixed_length_pairs_boxplot.png")

if __name__ == "__main__":
    num_boards = 10
    num_simulations = 10000
    num_snakes = 10
    num_ladders = 10

    length_pairs = [(5, 10), (10, 20), (20, 40), (40, 20), (20, 10), (10, 5)]

    simulate_fixed_length_pairs(num_boards, num_simulations, num_snakes, num_ladders, length_pairs)
