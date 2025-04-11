# Game Theory Hybrids
from .hybrid_ld_gwo import HybridLDGWO
from .hybrid_rd_gwo import HybridRDGWO
from .hybrid_fp_gwo import HybridFPGWO
from .hybrid_brd_gwo import HybridBRDGWO

# Metaheuristic Hybrids
from .hybrid_pso import HybridPSO
from .hybrid_ga import HybridGA
from .hybrid_woa import HybridWOA
from .hybrid_hho import HybridHHO
from .hybrid_ssa import HybridSSA
from .hybrid_ao import HybridAO
from .hybrid_rsa import HybridRSA
from .hybrid_tsa import HybridTSA
from .hybrid_gbo import HybridGBO
from .hybrid_avo import HybridAVO
from .hybrid_qana import HybridQANA
from .hybrid_pso import HybridPSO

__all__ = [
    # Game Theory (4)
    'HybridLDGWO',
    'HybridRDGWO',
    'HybridFPGWO',
    'HybridBRDGWO',

    # Metaheuristics (11)
    'HybridPSO',
    'HybridGA',
    'HybridWOA',
    'HybridHHO',
    'HybridSSA',
    'HybridAO',
    'HybridRSA',
    'HybridTSA',
    'HybridGBO',
    'HybridAVO',
    'HybridQANA'
    'HybridPSO',

]