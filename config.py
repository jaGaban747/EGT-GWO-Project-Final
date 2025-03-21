# config.py

# Problem Setup
NUM_TASKS = 50
NUM_EDGE_NODES = 10
MISSION_CRITICAL_RATIO = 0.2  # 20% of tasks are mission-critical

# Task properties (randomized in app.py)
TASK_CPU_RANGE = (1, 5)
TASK_DEADLINE_RANGE = (10, 50)
TASK_DATA_RANGE = (10, 100)

# Edge node properties (randomized in app.py)
EDGE_CPU_CAP_RANGE = (20, 40)
EDGE_ENERGY_COST_RANGE = (0.1, 0.5)

# Network properties
BANDWIDTH = 100  # Mbps

# Algorithm Hyperparameters
POP_SIZE = 30
MAX_ITER = 100
ALPHA = 0.5  # Weight for latency
GAMMA = 0.2  # Weight for energy
BETA = 0.7   # Rationality parameter for Logit Dynamics

# Random seed for reproducibility
RANDOM_SEED = 42