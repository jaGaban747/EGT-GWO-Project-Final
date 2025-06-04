# config.py

# ===============================================
# 📦 Task Offloading System Configuration for 6G MEC
# ===============================================

# -----------------------------------------------
# 📌 Problem Setup
# -----------------------------------------------

RELIABILITY_CRITICAL = 0.999  # Mission-critical reliability requirement
RELIABILITY_NORMAL = 0.95     # Normal task reliability requirement
NODE_FAILURE_PROB = 0.1       # Increased for RTSR variability
PACKET_LOSS_RATE = 0.02       # Increased for RTSR variability
ENERGY_SCALE = 1e-6           # For EACE
DISRUPTION_PROB = 0.15        # Increased for DAS variability
SCALE_FACTORS = [0.5, 1, 3, 5]  # Wider range for SI
BETA7 = 0.15  # EACE weight
BETA8 = 0.1   # SI weight
BETA9 = 0.1   # DAS weight
BETA10 = 0.2  # RTSR weight

# Number of computational tasks to offload
NUM_TASKS = 50

# Number of available edge servers/nodes
NUM_EDGE_NODES = 10

# Percentage of tasks that are mission-critical
MISSION_CRITICAL_RATIO = 0.2  # 20%

# -----------------------------------------------
# ⚙️ Task Properties
# -----------------------------------------------

# CPU cycles required for each task (in Millions of Instructions)
TASK_CPU_RANGE = (1, 5)  # 1 to 5 MIs

# Deadline range for each task (seconds)
TASK_DEADLINE_MIN = 0.05  # Tighter for RTSR, DAS
TASK_DEADLINE_MAX = 0.5   # Tighter for RTSR, DAS

DISTANCE_SCALE_FACTOR = 1000

# Data size associated with each task (in Megabytes)
TASK_DATA_RANGE = (10, 100)

# -----------------------------------------------
# 🖥️ Edge Node Properties
# -----------------------------------------------

# CPU capacity range for edge nodes (in MIPS)
EDGE_CPU_CAP_RANGE = (5, 15)

# Energy cost per task unit (arbitrary energy units)
EDGE_ENERGY_COST_RANGE = (0.1, 0.5)

# -----------------------------------------------
# 🌐 Network & 6G Environment Parameters
# -----------------------------------------------

# Bandwidth (in Mbps)
BANDWIDTH = 1000

# Latency bound for URLLC in 6G (seconds)
LATENCY_BOUND = 1e-3  # 1 millisecond

# Max supported device mobility speed (km/h)
MAX_MOBILITY_SPEED = 500

# Device density (per square km)
DEVICE_DENSITY = 1000000

# -----------------------------------------------
# 🧠 Algorithm Hyperparameters
# -----------------------------------------------

# Population size
POP_SIZE = 30

# Maximum iterations
MAX_ITER = 100

# Latency weight
ALPHA = 0.5

# Energy weight
GAMMA = 0.2

# Logit Dynamics rationality parameter
BETA = 0.7

# Q-learning parameters
EPSILON = 0.9
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9

# Strategy adjustment factor
DELTA = 0.1

# -----------------------------------------------
# 🔁 Reproducibility
# -----------------------------------------------

RANDOM_SEED = 42