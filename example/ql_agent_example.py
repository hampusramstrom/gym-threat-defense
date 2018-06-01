
import numpy as np
import random
import gym
import gym_threat_defense  # noqa

from ql_agent_parameters import STATES


def choose_action(env, state, Q, i):  # noqa
    eps = 1 - i * 0.001

    if random.uniform(0, 1) < eps:
        # SHOULD THIS BE POSSIBLE?
        # return env.action_space.sample()
        return random.randint(0, 3)
    else:
        return np.argmax(Q[state])


# MAKE AS HASH?
def get_index_in_matrix(observation):
    for i in range(STATES.shape[0]):
        if np.array_equal(observation, STATES[i]):
            return i


def q_learning(env):
    # NOT WORKING
    # Q = np.zeros([env.observation_space.n, env.action_space.n])
    Q = np.zeros([29, 4])  # noqa
    alpha = 0.85
    gamma = 0.95
    num_episodes = 2000

    rewards = []

    for i in range(num_episodes):
        s_list = env.reset()
        s = get_index_in_matrix(s_list)
        done = False
        j = 0
        r_all = 0

        # What should j be?
        while j < 99:
            j += 1
            a = choose_action(env, s, Q, i)
            # sn_list, r, done, _ = env.step(a)
            # sn = get_index_in_matrix(sn_list)
            sn_list, r, done, info = env.step(a)
            sn = get_index_in_matrix(info['state'])
            Q[s, a] = Q[s, a] + alpha * (r + gamma * np.max(Q[sn]) - Q[s, a])
            s = sn
            r_all += r

            if done:
                break

        rewards.append(r_all)
        if i % 100 == 0 and i > 0:
            print 'Episode: %s' % i
            print "Score over last 100 episodes: " + \
                str(sum(rewards[(i - 100):i]) / 100)

    print "Score over time: " + str(sum(rewards) / num_episodes)
    print "Final Q-Table Values"
    print Q


env = gym.make('threat-defense-v0')
q_learning(env)
