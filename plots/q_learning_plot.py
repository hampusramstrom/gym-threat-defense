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
    num_episodes = 1000
    n_simulations = 100

    rewards = np.zeros([n_simulations, num_episodes])

    for j in range(n_simulations):
        for i in range(num_episodes):
            o_list = env.reset()
            o = get_index_in_matrix(env, o_list)
            done = False
            j = 0
            r_all = 0

            while j < 99:
                j += 1
                a = choose_action(env, o, q, i)

                on_list, r, done, _ = env.step(a)
                on = get_index_in_matrix(env, on_list)
                q[o, a] = q[o, a] + alpha * (r + gamma * np.max(q[on]) - q[o, a])
                o = on
                r_all += r

                if done:
                    break

        rewards[j,i] = r_all

    print "Simulation:", j

    all_averages = np.mean(rewards, axis=0).tolist()
    stds = np.std(rewards, axis=0).tolist()

    with open('q_learning_res.csv', 'w') as f:
      writer = csv.writer(f, delimiter='\t')
      episode_numbers = ['E'] + range(1, num_episodes + 1)
      writer.writerows(zip(episode_numbers, ['A'] + all_averages))

    with open('q_learning_std_high.csv', 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        episode_numbers = ['E'] + range(1, num_episodes + 1)
        stds_up = map(lambda x: x[0] + x[1], zip(all_averages, stds))

        writer.writerows(zip(episode_numbers, ['V'] + stds_up))

    with open('q_learning_std_low.csv', 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        episode_numbers = ['E'] + range(1, num_episodes + 1)
        stds_up = map(lambda x: x[0] - x[1], zip(all_averages, stds))

        writer.writerows(zip(episode_numbers, ['V'] + stds_up))


env = gym.make('threat-defense-v0')
q_learning(env)
