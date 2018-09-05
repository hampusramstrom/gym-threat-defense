"""
A simple example on how to use the Threat Defense environment,
applying Q-learning where a table is used to store the data.

Authors:
Johan Backman - johback@student.chalmers.se
Hampus Ramstrom - hampusr@student.chalmers.se
"""

import numpy as np
import random
import gym
import gym_threat_defense  # noqa


def choose_action(env, observation, q, i):  # noqa
    """
    Chooses a new action, either randomly or the one with
    max value in the Q table, depending on the amount of randomness.

    Arguments:
    env -- the Threat Defense gym environment.
    observation -- an observation as its numeric index in the states matrix,
        containing all states.
    q -- the Q table, containing the data.
    i -- the current episode in the simulation.

    Returns:
    An action containing a numeric value [0, 3].
    """
    dec = 0.01
    eps = 1 - i * dec

    if random.uniform(0, 1) < eps:
        return env.action_space.sample()
    else:
        return np.argmax(q[observation])


def get_index_in_matrix(env, observation):
    """
    Retrieves the index of an observation in the STATES matrix,
    containing all states.

    Arguments:
    env -- the Threat Defense gym environment.
    observation -- an observation as a binary vector of length 12.

    Returns:
    A numeric index.
    """
    for i in range(env.all_states.shape[0]):
        if np.array_equal(observation, env.all_states[i]):
            return i


def q_learning(env):
    """
    Runs Q-learning with a simple table for storing the data and prints the
mean reward for the last 100 episodes as well as printing the Q-table at the
    end of the simulation.

    Arguments:
    env -- the Threat Defense gym environment.
    """
    q = np.zeros([env.observation_space.n, env.action_space.n])
    alpha = 0.1
    gamma = 0.7
    num_episodes = 2000

    rewards = []

    for i in range(num_episodes):
        o_list = env.reset()
        o = get_index_in_matrix(env, o_list)
        done = False
        r_all = 0

        while True:
            a = choose_action(env, o, q, i)

            on_list, r, done, _ = env.step(a)
            on = get_index_in_matrix(env, on_list)
            q[o, a] = q[o, a] + alpha * (r + gamma * np.max(q[on]) - q[o, a])
            o = on
            r_all += r

            if done:
                break

        rewards.append(r_all)
        if i % 100 == 0 and i > 0:
            print 'Episode: %s' % i
            print "Score over the last 100 episodes: " + \
                str(sum(rewards[(i - 100):i]) / 100)

    print "Score over time: " + str(sum(rewards) / num_episodes)
    print "Final Q-Table values"
    print q


env = gym.make('threat-defense-v0')
q_learning(env)
