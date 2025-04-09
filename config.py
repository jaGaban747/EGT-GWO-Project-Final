# config.py

# ===============================================
# 📦 Task Offloading System Configuration for 6G MEC
# ===============================================

# -----------------------------------------------
# 📌 Problem Setup
# -----------------------------------------------

# Number of computational tasks to offload
NUM_TASKS = 50

# Number of available edge servers/nodes
NUM_EDGE_NODES = 10

# Percentage of tasks that are mission-critical (requiring stricter latency)
MISSION_CRITICAL_RATIO = 0.2  # 20%

# -----------------------------------------------
# ⚙️ Task Properties
# -----------------------------------------------

# CPU cycles required for each task (in Millions of Instructions)
TASK_CPU_RANGE = (1, 5)  # e.g., 1 to 5 MIs

# Deadline range for each task (arbitrary time units)
TASK_DEADLINE_RANGE = (10, 50)

# Data size associated with each task (in Megabytes)
TASK_DATA_RANGE = (10, 100)

# -----------------------------------------------
# 🖥️ Edge Node Properties
# -----------------------------------------------

# CPU capacity range for edge nodes (in MIPS: Million Instructions Per Second)
EDGE_CPU_CAP_RANGE = (20, 40)

# Energy cost per task unit (arbitrary energy units)
EDGE_ENERGY_COST_RANGE = (0.1, 0.5)

# -----------------------------------------------
# 🌐 Network & 6G Environment Parameters
# -----------------------------------------------

# Bandwidth (in Mbps); can scale this for 6G-level throughput
BANDWIDTH = 1000  # Consider increasing this for 6G (e.g., 1000 to 10000)

# Latency bounds for ultra-reliable low-latency communication (URLLC) in 6G
LATENCY_BOUND = 1e-3  # 1 millisecond

# Max supported device mobility speed (e.g., 500 km/h in 6G)
MAX_MOBILITY_SPEED = 500  # in km/h

# Device density in high-load scenarios (per square km)
DEVICE_DENSITY = 1000000  # 1 million devices/km² for 6G

# -----------------------------------------------
# 🧠 Algorithm Hyperparameters
# -----------------------------------------------

# Population size for population-based algorithms (e.g., GWO, PSO)
POP_SIZE = 30

# Maximum number of iterations for convergence
MAX_ITER = 100

# Latency weight in fitness calculation
ALPHA = 0.5

# Energy weight in fitness calculation
GAMMA = 0.2

# Rationality parameter for Logit Dynamics (LD-GWO)
BETA = 0.7

# Q-learning parameters
EPSILON = 0.9          # Initial exploration rate
LEARNING_RATE = 0.1    # Alpha - learning rate
DISCOUNT_FACTOR = 0.9  # Gamma - discount factor

# Strategy adjustment factor (used in stochastic updates or decay)
DELTA = 0.1

# -----------------------------------------------
# 🔁 Reproducibility
# -----------------------------------------------

# Global random seed to ensure consistent results
RANDOM_SEED = 42
