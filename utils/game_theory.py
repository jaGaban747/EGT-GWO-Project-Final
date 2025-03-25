import numpy as np
from config import BETA

def logit_dynamics(utilities):
    exp_utilities = np.exp(BETA * utilities)
    return exp_utilities / np.sum(exp_utilities)