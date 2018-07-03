
import numpy as np
import random
import gym
import gym_threat_defense  # noqa


def choose_action(env, state, q, i, inc):  # noqa
    eps = 1 - i * inc

    if random.uniform(0, 1) < eps:
        return env.action_space.sample()
    else:
        return np.argmax(q[state])


def get_index_in_matrix(env, observation):
    for i in range(env.all_states.shape[0]):
        if np.array_equal(observation, env.all_states[i]):
            return i


def q_learning(env):
    incs = np.array([0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1])
    results = np.zeros([len(incs)])
    ind = -1
    for inc in incs:
        ind += 1
        print "inc:", inc
        q = np.zeros([env.observation_space.n, env.action_space.n])
        alpha = 0.1
        gamma = 0.7
        num_episodes = 2000

        rewards = []

        for i in range(num_episodes):
            s_list = env.reset()
            s = get_index_in_matrix(env, s_list)
            done = False
            j = 0
            r_all = 0

            # What should j be?
            while j < 99:
                j += 1
                a = choose_action(env, s, q, i, inc)
                sn_list, r, done, _ = env.step(a)
                sn = get_index_in_matrix(env, sn_list)
                # sn_list, r, done, info = env.step(a)
                # sn = get_index_in_matrix(info['state'])
                q[s, a] = q[s, a] + alpha * \
                    (r + gamma * np.max(q[sn]) - q[s, a])
                s = sn
                r_all += r

                if done:
                    break

            rewards.append(r_all)
        results[ind] = sum(rewards[1000:])
    print "The results array:"
    print results

    print "The maximal indexes:", \
        np.unravel_index(np.argmax(results, axis=None), results.shape)


env = gym.make('threat-defense-v0')
q_learning(env)
