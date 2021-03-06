

import numpy as np
import random
import gym
import gym_threat_defense  # noqa


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

    num_episodes = 2000

    rewards = []

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

        rewards.append(r_all)
        if i % 100 == 0 and i > 0:
            print 'Episode: %s' % i
            print "Score over the last 100 episodes: " + \
                str(sum(rewards[(i - 100):i]) / 100)
    print "Score over time: " + str(sum(rewards) / num_episodes)


env = gym.make('threat-defense-v0')
optimal(env)
