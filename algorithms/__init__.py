# Game Theory Hybrids
from .hybrid_ld_gwo import HybridLDGWO
from .hybrid_rd_gwo import HybridRDGWO
from .hybrid_fp_gwo import HybridFPGWO
from .hybrid_brd_gwo import HybridBRDGWO

# Metaheuristic Hybrids
from .hybrid_pso import HybridPSO
from .hybrid_ga import HybridGA

# Reinforcement Learning-Based Hybrids
#from .hybrid_ddqn import HybridDDQN
#from .hybrid_ddpg import HybridDDPG
from .hybrid_qlearning import HybridQlearning
from .hybrid_nsga2 import HybridNSGA2

# Scheduling Hybrids
from .hybrid_heft import HybridHEFT
from .hybrid_maxmin import HybridMaxMin
from .hybrid_lyapunov import HybridLyapunov

__all__ = [
    # Game Theory
    'HybridLDGWO',
    'HybridRDGWO',
    'HybridFPGWO',
    'HybridBRDGWO',

    # Metaheuristics
    'HybridPSO',
    'HybridGA',

    # Reinforcement Learning
    'HybridDDQN',
    'HybridDDPG',
    'HybridQlearning',

    # Scheduling Algorithms
    'HybridHEFT',
    'HybridMaxMin',
    'HybridLyapunov',
]
