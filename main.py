import numpy as np
import pandas as pd
import sys
import ast
from maxent import maxent, optimizer
from helpers import get_trajectories, get_transition_probability

feature_matrix = pd.read_csv('reference_states.csv', header=None).values
feature_matrix[feature_matrix.shape[0] - 1] = [0] * feature_matrix.shape[1]
feature_matrix = np.array(feature_matrix)

sas_transition = get_transition_probability()
ssa_transition = 0
trajectories = get_trajectories()

reward = []
with open('reward_1.0.txt', 'r') as r:
    reward = np.array(ast.literal_eval(r.read()))

t = np.linalg.solve(feature_matrix[250:400,:], reward[250:400])

optim = optimizer.Sga(lr=0.01)
init = optimizer.Constant(1)
terminal = [feature_matrix.shape[0] - 1]
r = maxent.irl(ssa_transition, sas_transition, feature_matrix, [feature_matrix.shape[0] - 1], trajectories, optim, init)
# r = maxent.irl_causal(ssa_transition, sas_transition, feature_matrix, terminal, trajectories,
#                       optim, init, 1)
