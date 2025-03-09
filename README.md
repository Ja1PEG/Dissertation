# Snakes and Ladders Simulation Project
## Description
A Python-based statistical analysis of Snakes and Ladders game mechanics, exploring the impact of snake and ladder counts and length distributions on game completion times.

## Project Overview
This project simulates and analyzes the Snakes and Ladders game using various configurations:

### Chapter 1: Investigates the effect of different numbers of snakes and ladders on game length, keeping snake and ladder lengths constant.
### Chapter 2: Explores the influence of different probability distributions (Uniform, Normal, Exponential) for snake/ladder lengths on game completion times.

### Project Structure
```
├── Chapter 1/
│   ├── CSV/
│   │   └── Run 1/
│   │       └── Count Variation/
│   │           └── ... (CSV files with game results)
│   ├── Chapter1Writing.pdf
│   ├── Chapter1Writing.tex
│   ├── GameTimeTrend.py
│   └── Snakes and Ladders.py
├── Chapter 2/
│   ├── TestingDir/
│   │   ├── final_results.csv
│   │   └── boxplotgen.py
│   │       └── ... (folders with CSV files for different snake/ladder counts)
│   ├── withLength/
│   │   ├── Discards/
│   │   │   └── ... (folders with discarded simulation runs)
│   │   ├── FinalSampling/
│   │   │   └── ... (CSV files and Test.py for final sampling analysis)
│   │   ├── RandomLength/
│   │   │   └── ... (files for random length analysis)
│   │   └── UnequalLengths/
│   │       └── ... (files for unequal lengths analysis)
│   └── Writing/
│       └── ... (PDF and LaTeX files for Chapter 2 writing)
├── Combined Chapter/
│   ├── Chapter1.log
│   ├── Chapter1.tex
│   ├── Chapter2.tex
│   ├── Chapter3.tex
│   ├── Combined_Ch1_Ch2.log
│   ├── Combined_Ch1_Ch2.pdf
│   ├── Combined_Ch1_Ch2.tex
│   └── OtherPages/
│       └── ... (LaTeX files for title page, acknowledgements, etc.)
└── IntroChap/
    ├── Introduction.log
    ├── Introduction.pdf
    └── Introduction.tex
```

### Features
- Board Generation:
  - Configurable number of snakes and ladders
  - Variable length distribution (Uniform, Normal, Exponential)
  - Collision prevention logic
- Simulation:
  - Multiple game simulations for each configuration
  - Tracking of game completion times
- Analysis:
  - Statistical analysis of game completion times
  - Generation of box plots and other visualizations
  - Comparison of results across different configurations


### Dependencies
- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn

### Usage
- Install Dependencies: Use `pip install numpy pandas matplotlib seaborn`
- Run Simulations: Execute the Python scripts in the respective chapter folders (e.g., `Chapter 1/Snakes and Ladders.py`)
- Analyze Results: Use the provided scripts (e.g., `Chapter 2/TestingDir/boxplotgen.py`) or your own methods to analyze the generated CSV data.

### Contributing
Feel free to fork the repository and submit pull requests for any improvements or additional analyses.