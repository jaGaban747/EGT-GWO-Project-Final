# Game Theory Hybrids
from .hybrid_ld_gwo import HybridLDGWO
from .hybrid_rd_gwo import HybridRDGWO
from .hybrid_fp_gwo import HybridFPGWO
from .hybrid_brd_gwo import HybridBRDGWO

# Metaheuristic Hybrids
from .hybrid_pso import HybridPSO
from .hybrid_coa import HybridCOA
from .hybrid_woa import HybridWOA
from .hybrid_hho import HybridHHO
from .hybrid_ssa import HybridSSA
from .hybrid_ao import HybridAO
from .hybrid_rsa import HybridRSA
from .hybrid_tsa import HybridTSA
from .hybrid_do import HybridDO
from .hybrid_avo import HybridAVO
from .hybrid_sho import HybridSHO
from .hybrid_pso import HybridPSO
from .hybrid_gto import HybridGTO

__all__ = [
    # Game Theory (4)
    'HybridLDGWO',
    'HybridRDGWO',
    'HybridFPGWO',
    'HybridBRDGWO',

    # Metaheuristics (11)
    'HybridPSO',
    'HybridCOA',
    'HybridWOA',
    'HybridHHO',
    'HybridSSA',
    'HybridAO',
    'HybridRSA',
    'HybridTSA',
    'HybridDO',
    'HybridAVO',
    'HybridSHO'
    'HybridPSO',
    'HybridGTO',

]