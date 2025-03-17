import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BOARD_SIZE = 100

# Generate a board with random start and end points
def generate_board_random_points(num_snakes, num_ladders):
    """
    Generates a board with randomised start and end points for snakes and ladders.
    """
    snakes_and_ladders = {}
    snake_positions = []
    ladder_positions = []

    # Assign snakes
    for _ in range(num_snakes):
        while True:
            start = random.randint(2, BOARD_SIZE - 1)
            end = random.randint(1, start - 1)
            if start not in snakes_and_ladders and end not in snakes_and_ladders.values():
                snakes_and_ladders[start] = end
                snake_positions.append((start, end))
                break

    # Assign ladders
    for _ in range(num_ladders):
        while True:
            start = random.randint(2, BOARD_SIZE - 1)
            end = random.randint(start + 1, BOARD_SIZE)
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

# Simulate games, log data, and generate chart
def simulate_random_points(num_boards, num_simulations, num_snakes, num_ladders):
    results = []
    board_details = []

    for board_num in range(num_boards):
        snakes_and_ladders, snake_positions, ladder_positions = generate_board_random_points(num_snakes, num_ladders)
        game_times = [play_game(snakes_and_ladders) for _ in range(num_simulations)]
        avg_time = np.mean(game_times)
        results.append(avg_time)

        # Log board details
        board_details.append({
            "Board Number": f"Board {board_num + 1}",
            "Snakes": "; ".join([f"{s[0]}->{s[1]}" for s in snake_positions]),
            "Ladders": "; ".join([f"{l[0]}->{l[1]}" for l in ladder_positions]),
            "Average Game Time": avg_time
        })

    # Save board configurations and results to CSV
    board_df = pd.DataFrame(board_details)
    board_df.to_csv("approach_3_board_details.csv", index=False)

    # Plot results
    plt.figure(figsize=(10, 6))
    sns.barplot(x=[f"Board {i+1}" for i in range(num_boards)], y=results, palette="Greens_d")
    # Add values to the top of the bar plots
    for i, v in enumerate(results):
        plt.text(i, v + 0.5, f"{v:.2f}", ha='center', va='bottom')
    plt.title("Average Game Duration for Random Points")
    plt.xlabel("Board Number")
    plt.ylabel("Average Game Duration (Turns)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("approach_3_random_points.png")
    
        # Plot frequency distribution of game times for the last board
    plt.figure(figsize=(10, 6))
    sns.histplot(game_times, bins=30, kde=True, color='blue')
    plt.title("Frequency Distribution of Game Duration for the Last Board")
    plt.xlabel("Game Duration (Turns)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("approach_3_game_time_distribution.png")
    
if __name__ == "__main__":
    num_boards = 10
    num_simulations = 1000
    num_snakes = 10
    num_ladders = 10

    simulate_random_points(num_boards, num_simulations, num_snakes, num_ladders)
