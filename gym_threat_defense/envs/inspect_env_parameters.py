"""Inspect env states, rewards and transitions."""

import numpy as np


STATES = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    ####################################
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    ####################################
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0],
    ####################################
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
    [1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
    [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
    [1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    ####################################
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
])

REWARDS = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
    [-0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2,
    -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2,
    -0.2, -0.2, -0.2, -0.2, -1.2],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
     -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
     -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
    [-5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5,
     -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -6],
])


def transition_matrix():
    """TODO docstring."""
    none = np.array([
        [0.25, 0.25, 0.25, 0.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0.1, 0, 0.1, 0.4, 0.4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0.1, 0.1, 0, 0, 0.4, 0.4, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0.04, 0, 0.16, 0, 0.16, 0.64, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ###########################################################
        [0, 0, 0, 0, 0.1, 0.1, 0, 0, 0, 0.4, 0.4, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0.04, 0, 0, 0.16, 0, 0.16, 0.64, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0.05, 0.05, 0, 0, 0, 0, 0.45, 0.45, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0.02, 0.08, 0, 0, 0, 0, 0.18, 0.72,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0.02, 0, 0, 0.08, 0, 0, 0.18,
         0.72, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ###########################################################
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.05, 0.05, 0, 0, 0, 0, 0,
         0.45, 0.45, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.02, 0.08, 0, 0, 0, 0, 0,
         0.18, 0.72, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01, 0, 0, 0, 0.09, 0,
         0, 0.09, 0.81, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.1, 0, 0, 0, 0,
         0, 0, 0.4, 0.4, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.04, 0.16, 0, 0,
         0, 0, 0, 0, 0.16, 0.64, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.04, 0.16, 0,
         0, 0, 0, 0, 0, 0.16, 0.64, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.02, 0, 0,
         0, 0.18, 0, 0, 0, 0.08, 0.72, 0, 0, 0, 0],
        ###########################################################
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2,
         0.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0.1, 0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0.2, 0, 0, 0, 0, 0.8, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0.2, 0.8, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0.2, 0.8, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0.1, 0.9, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0.36, 0.64, 0, 0, 0],
        ###########################################################
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0.1, 0.9, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0.36, 0.64, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.9],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    ])

    inspect = none

    block = np.array([
        [0.5, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.5, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0.2, 0, 0, 0, 0.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0.2, 0, 0, 0, 0.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ###########################################################
        [0.5, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0.2, 0, 0, 0, 0.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0.9, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0.9, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0.9, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ###########################################################
        [0.5, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0.2, 0, 0, 0, 0.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0.9, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0, 0, 0, 0, 0, 0,
         0, 0.8, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0, 0, 0, 0, 0, 0,
         0, 0.8, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0, 0, 0, 0, 0, 0,
         0, 0.8, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0, 0, 0, 0, 0, 0,
         0, 0.8, 0, 0, 0, 0, 0, 0, 0, 0],
        ###########################################################
        [0.5, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0.2, 0, 0, 0, 0.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0.9, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0, 0, 0, 0, 0, 0,
         0, 0.8, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        ###########################################################
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    ])

    disconnect = np.array([
        [0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0.2, 0, 0, 0.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0.2, 0, 0, 0.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ###########################################################
        [0, 0, 0, 0, 0.2, 0, 0, 0, 0, 0.8, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0.2, 0, 0, 0, 0, 0.8, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0.2, 0, 0, 0.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0.2, 0, 0, 0, 0, 0.8, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ###########################################################
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0.9, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0.9, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0.9, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0.2, 0, 0, 0.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0.2, 0, 0, 0, 0, 0.8, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0.9, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ###########################################################
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0.2, 0, 0, 0.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0.2, 0, 0, 0, 0, 0.8, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0.9, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ###########################################################
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])

    both = np.zeros(29, dtype=float)
    both[0] = 1
    both = np.tile(both, (29, 1))

    return np.array([none, inspect, block, disconnect, both])


TRANSITIONS = transition_matrix()


"""
The observation matrix, i.e. the probability of recieving
observation o when standing i state s.
"""
OBSERVATIONS = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0],
    [0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0],
    [0.5, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0],
    [0.25, 0.25, 0.25, 0.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ###################################################################
    [0.25, 0.25, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.125, 0.125, 0.125, 0.125, 0.25, 0.25, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.25, 0, 0.25, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.125, 0.125, 0.125, 0.125, 0, 0, 0.25, 0.25, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0625, 0.0625, 0.0625, 0.0625, 0.125, 0.125, 0.125, 0.125, 0.25,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ###################################################################
    [0.125, 0.125, 0, 0, 0.25, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0625, 0.0625, 0.0625, 0.0625, 0.125, 0.125, 0, 0, 0, 0.25, 0.25,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.03125, 0.03125, 0.03125, 0.03125, 0.0625, 0.0625, 0.0625,
     0.0625, 0.125, 0.125, 0.125, 0.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0],
    [0.125, 0, 0.125, 0, 0, 0, 0.25, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0625, 0.0625, 0.0625, 0.0625, 0, 0, 0.125, 0.125, 0, 0, 0, 0,
     0.25, 0.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.03125, 0.03125, 0.03125, 0.03125, 0.0625, 0.0625, 0.0625,
     0.0625, 0.125, 0, 0, 0, 0.125, 0.125, 0.25, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0],
    [0.015625, 0.015625, 0.015625, 0.015625, 0.03125, 0.03125, 0.03125,
     0.03125, 0.0625, 0.0625, 0.0625, 0.125, 0.0625, 0.0625, 0.125,
     0.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ###################################################################
    [0.0625, 0.0625, 0, 0, 0.125, 0, 0, 0, 0, 0.25, 0, 0, 0, 0, 0, 0,
     0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.03125, 0.03125, 0.03125, 0.03125, 0.0625, 0.0625, 0, 0, 0,
     0.125, 0.125, 0, 0, 0, 0, 0, 0.25, 0.25, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0],
    [0.015625, 0.015625, 0.015625, 0.015625, 0.03125, 0.03125, 0.03125,
     0.03125, 0.0625, 0.0625, 0.0625, 0.125, 0, 0, 0, 0, 0.125, 0.125,
     0.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.015625, 0.015625,
     0.015625, 0.015625, 0.03125, 0.03125, 0.03125, 0.0625, 0.03125,
     0.03125, 0.0625, 0.125, 0.0625, 0.0625, 0.125, 0.25, 0, 0, 0, 0,
     0, 0, 0, 0, 0],
    [0.0375, 0, 0.0375, 0, 0, 0, 0.075, 0, 0, 0, 0, 0, 0.15, 0, 0, 0,
     0, 0, 0, 0, 0.7, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.01875, 0.01875, 0.01875, 0.01875, 0, 0, 0.0375, 0.0375, 0, 0, 0,
     0, 0.075, 0.075, 0, 0, 0, 0, 0, 0, 0.35, 0.35, 0, 0, 0, 0, 0, 0,
     0],
    [0.009375, 0.009375, 0.009375, 0.009375, 0.01875, 0.01875, 0.01875,
     0.01875, 0.0375, 0, 0, 0, 0.0375, 0.0375, 0.075, 0, 0, 0, 0, 0,
     0.175, 0.175, 0.35, 0, 0, 0, 0, 0, 0],
    [0.0046875, 0.0046875, 0.0046875, 0.0046875, 0.009375, 0.009375,
     0.009375, 0.009375, 0.01875, 0.01875, 0.01875, 0.0375, 0.01875,
     0.01875, 0.0375, 0.075, 0, 0, 0, 0, 0.0875, 0.0875, 0.175, 0.35,
     0, 0, 0, 0, 0],
    [0.00234375, 0.00234375, 0.00234375, 0.00234375, 0.0046875,
     0.0046875, 0.0046875, 0.0046875, 0.009375, 0.009375, 0.009375,
     0.01875, 0.009375, 0.009375, 0.01875, 0.0375, 0.01875, 0.01875,
     0.0375, 0.075, 0.04375, 0.04375, 0.0875, 0.175, 0.35, 0, 0,
     0, 0],
    ###################################################################
    [0.0009375, 0.0009375, 0.0009375, 0.0009375, 0.001875, 0.001875,
     0.001875, 0.001875, 0.00375, 0.00375, 0.00375, 0.0075, 0.00375,
     0.00375, 0.0075, 0.015, 0.0075, 0.0075, 0.015, 0.03, 0.0175,
     0.0175, 0.035, 0.07, 0.14, 0.6, 0, 0, 0],
    [0.00028125, 0.00028125, 0.00028125, 0.00028125, 0.0005625,
     0.0005625, 0.0005625, 0.0005625, 0.001125, 0.001125, 0.001125,
     0.00225, 0.001125, 0.001125, 0.00225, 0.0045, 0.00225, 0.00225,
     0.0045, 0.009, 0.00525, 0.00525, 0.0105, 0.021, 0.042, 0.18, 0.7,
     0, 0],
    [0.0000421875, 0.0000421875, 0.0000421875, 0.0000421875,
     0.000084375, 0.000084375, 0.000084375, 0.000084375, 0.00016875,
     0.00016875, 0.00016875, 0.0003375, 0.00016875, 0.00016875,
     0.0003375, 0.000675, 0.0003375, 0.0003375, 0.000675, 0.00135,
     0.0007875, 0.0007875, 0.001575, 0.00315, 0.0063, 0.027, 0.105,
     0.85, 0],
    [0.000002109375, 0.000002109375, 0.000002109375, 0.000002109375,
     0.00000421875, 0.00000421875, 0.00000421875, 0.00000421875,
     0.0000084375, 0.0000084375, 0.0000084375, 0.000016875,
     0.0000084375, 0.0000084375, 0.000016875, 0.00003375, 0.000016875,
     0.000016875, 0.00003375, 0.0000675, 0.000039375, 0.000039375,
     0.00007875, 0.0001575, 0.000315, 0.00135, 0.00525, 0.0425, 0.95]
])
