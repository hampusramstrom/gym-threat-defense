

import numpy as np
import random
import gym
import gym_threat_defense  # noqa

# ONLY FOR RESULT PLOTTING
import csv


def choose_action(env, s):
    if s == env.state_space.n - 2:
        if random.randint(0, 1) == 0:
            return 1
        else:
            return 2
    else:
        return 0


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


def optimal(env):
    n_simulations = 100
    num_episodes = 1000

    rewards = np.zeros([n_simulations, num_episodes])

    for j in range(n_simulations):
        for i in range(num_episodes):
            s_list = env.reset()
            s = get_index_in_matrix(env, s_list)
            done = False
            r_all = 0

            while True:
                a = choose_action(env, s)
                _, r, done, info = env.step(a)
                s_list = info['state']
                s = get_index_in_matrix(env, s_list)

                r_all += r

                if done:
                    break

            rewards[j,i] = r_all

        print "Simulation:", j

    with open('optimal_res.csv', 'w') as f:
      writer = csv.writer(f, delimiter='\t')
      episode_numbers = ['E'] + range(1, num_episodes + 1)
      writer.writerows(zip(episode_numbers, ['A'] + all_averages))

    with open('optimal_res_std_high.csv', 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        episode_numbers = ['E'] + range(1, num_episodes + 1)
        stds_up = map(lambda x: x[0] + x[1], zip(all_averages, stds))

        writer.writerows(zip(episode_numbers, ['V'] + stds_up))

    with open('optimal_res_std_low.csv', 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        episode_numbers = ['E'] + range(1, num_episodes + 1)
        stds_up = map(lambda x: x[0] - x[1], zip(all_averages, stds))

        writer.writerows(zip(episode_numbers, ['V'] + stds_up))

env = gym.make('threat-defense-v0')
optimal(env)