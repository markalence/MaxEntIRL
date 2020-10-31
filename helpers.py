import ast
import numpy as np
import pandas as pd
from maxent import trajectory

actions = \
    [[0, 0, 0, 0],
     [0, 0, 0, 1],
     [0, 0, 1, 0],
     [0, 0, 1, 1],
     [0, 1, 0, 0],
     [0, 1, 0, 1],
     [1, 0, 0, 0],
     [1, 0, 1, 0],
     [1, 1, 0, 0]]
states = pd.read_csv('reference_states.csv', header=None)
S = len(states)
T = len(actions)


def get_transition_probability():
    df = pd.DataFrame()
    for i in range(159):
        df = df.append(pd.read_json(f'Trajectories/{i}.json', lines=True))
    p_transition = np.array([[[0 for i in range(S)] for j in range(T)] for k in range(S)])
    p_transition = p_transition.astype(float)
    p_mass = np.zeros((S, T))
    for i in range(len(df)):
        state_i = df.iloc[i].state_i
        a = df.iloc[i].action
        state_j = df.iloc[i].state_j
        p_mass[state_i][a] += 1
        p_transition[state_i][a][state_j] += 1
    print(p_mass)
    for i in range(S):
        for j in range(T):
            if p_mass[i][j] != 0:
                p_transition[i][j] /= p_mass[i][j]
    return np.array(p_transition)


def get_trajectories():
    t = []
    for i in range(159):
        df = pd.read_json(f'Trajectories/{i}.json', lines=True)
        tj = trajectory.Trajectory([(a, b, c) for a, b, c in df.values])
        t.append(tj)
    return np.array(t)
