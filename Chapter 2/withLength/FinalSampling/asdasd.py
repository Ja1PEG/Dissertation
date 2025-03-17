import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

BOARD_SIZE = 100
NUM_SIMULATIONS = 10000
NUM_SNAKES = 10
NUM_LADDERS = 10
MAX_LENGTH = 40

def sample_length_with_fixed_max(start_pos, max_length, distribution, is_snake=True):
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

        if length < 1 or length > max_length:
            continue

        end_pos = start_pos - length if is_snake else start_pos + length
        if 1 <= end_pos <= BOARD_SIZE:
            return length

def generate_board_with_sampling(num_snakes, num_ladders, max_length, distribution):
    snakes_and_ladders = {}
    while len(snakes_and_ladders) < (num_snakes + num_ladders):
        is_snake = len(snakes_and_ladders) < num_snakes
        start_pos = random.randint(2, BOARD_SIZE - 1)

        length = sample_length_with_fixed_max(start_pos, max_length, distribution, is_snake)
        end_pos = start_pos - length if is_snake else start_pos + length

        if start_pos in snakes_and_ladders or end_pos in snakes_and_ladders.values():
            continue

        snakes_and_ladders[start_pos] = end_pos

    return snakes_and_ladders

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

def simulate_games(snakes_and_ladders, num_simulations):
    return [play_game(snakes_and_ladders) for _ in range(num_simulations)]

# Generate and simulate boards for each distribution
distributions = ["exponential", "normal", "uniform"]
titles = ["Exponential", "Normal", "Uniform"]

for dist, title in zip(distributions, titles):
    board = generate_board_with_sampling(NUM_SNAKES, NUM_LADDERS, MAX_LENGTH, dist)
    game_times = simulate_games(board, NUM_SIMULATIONS)

    plt.figure(figsize=(8, 6))
    sns.histplot(game_times, bins=30, kde=True, color="blue")
    plt.title(f"Game Duration Distribution for Fixed Layout ({title})")
    plt.xlabel("Game Duration (Turns)")
    plt.ylabel("Frequency")
    
    filename = f"game_time_distribution_{dist}.png"
    plt.savefig(filename)
    plt.show()
