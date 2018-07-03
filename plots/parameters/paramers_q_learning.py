
import numpy as np
import random
import gym
import gym_threat_defense # noqa


def choose_action(env, state, q, i):  # noqa
    eps = 1 - i * 0.001

    if random.uniform(0, 1) < eps:
        return env.action_space.sample()
    else:
        return np.argmax(q[state])


def get_index_in_matrix(env, observation):
    for i in range(env.all_states.shape[0]):
        if np.array_equal(observation, env.all_states[i]):
            return i


def q_learning(env):
    results = np.zeros([len(np.arange(0, 1.01, 0.1)),
                        len(np.arange(0, 1.01, 0.1))])
    for alpha in np.arange(0, 1.01, 0.1):
        for gamma in np.arange(0, 1.01, 0.1):
            print "alpha:", alpha
            print "gamma:", gamma
            q = np.zeros([env.observation_space.n, env.action_space.n])
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
                    a = choose_action(env, s, q, i)
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

            results[int(10*alpha), int(10*gamma)] = sum(rewards[1000:])

    np.set_printoptions(threshold='nan')

    print "The results matrix:"
    print results

    print "The maximal indexes:", \
        np.unravel_index(np.argmax(results, axis=None), results.shape)


env = gym.make('threat-defense-v0')
q_learning(env)
