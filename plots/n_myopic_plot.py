

import numpy as np
import gym
import gym_threat_defense  # noqa

# ONLY FOR RESULT PLOTTING
import csv

def choose_action(env, i, o_list, r, a, emp_inspect, opt_a, eps):  # noqa
    if i < eps:
        if (a == 1 and
           get_index_in_matrix(env, o_list[1]) == env.state_space.n - 2):
            return np.random.randint(2, env.action_space.n - 1)
        else:
            return 1
    else:
        ind_o = get_index_in_matrix(env, o_list[0])
        # Best states are [27, 28].
        if np.argmax(emp_inspect[ind_o]) in [27, 28]:
            return opt_a
        else:
            return 0


def get_index_in_matrix(env, observation):
    for i in range(env.all_states.shape[0]):
        if np.array_equal(observation, env.all_states[i]):
            return i


def n_myopic(env):
    num_episodes = 1000
    n_simulations = 100
    time = np.zeros([n_simulations, num_episodes])
    rewards = np.zeros([n_simulations, num_episodes])

    for k in range(n_simulations):
        emp_inspect = np.zeros([env.observation_space.n, env.state_space.n])
        emp_inspect[env.observation_space.n - 1, env.state_space.n - 1] = 1

        eps = 10

        action_steps = []
        action_steps.append((4,))

        action_steps_matrix = np.zeros([env.action_space.n - 3, 100*eps])
        action_steps_ind = np.zeros(env.action_space.n - 3, dtype=int)
        average_len_of_a = np.zeros(env.action_space.n - 3)
        action_costs = [0, -0.2, -1, -1, -5]
        opt_a = 0

        for i in range(num_episodes):
            o_list = env.reset()
            done = False
            r_all = 0
            old_r = 0
            old_a = 0

            t_since_a = 0
            last_a = env.action_space.n

            l = 0

            while True:
                a = choose_action(env, i, o_list, old_r, old_a, emp_inspect,
                                  opt_a, eps)
                l += 1
                o_list, r, done, _ = env.step(a)
                if a == 1:
                    ind_o = get_index_in_matrix(env, o_list[0])
                    ind_s = get_index_in_matrix(env, o_list[1])
                    emp_inspect[ind_o, ind_s] += 1
                r_all += r
                old_r = r
                old_a = a

                t_since_a += 1

                if i < eps and a > 1:
                    if last_a < env.action_space.n:
                        action_steps_matrix[last_a - 2, action_steps_ind[last_a - 2]] \
                            = t_since_a
                        action_steps_ind[last_a - 2] += 1
                    t_since_a = 0
                    last_a = a
                elif i == eps:
                    for j in range(len(action_steps_matrix)):
                        if np.count_nonzero(action_steps_matrix[j]) == 0:
                            average_len_of_a[j] = np.sum(action_steps_matrix[j]) / \
                            abs(action_costs[j + 2])
                        else:
                            average_len_of_a[j] = (np.sum(action_steps_matrix[j]) / \
                                np.count_nonzero(action_steps_matrix[j])) / \
                                abs(action_costs[j + 2])
                    opt_a = np.argmax(average_len_of_a) + 2

                if done:
                    break

            rewards[k,i] = r_all
            time[k,i] = l

        print "Simulation:", k

    reward_averages = np.mean(rewards, axis=0).tolist()
    time_averages = np.mean(time, axis=0).tolist()
    reward_stds = np.std(rewards, axis=0).tolist()
    time_stds = np.std(time, axis=0).tolist()

    with open('n_myopic_n_0_eps_10_res.csv', 'w') as f:
      writer = csv.writer(f, delimiter='\t')
      episode_numbers = ['E'] + range(1, num_episodes + 1)
      writer.writerows(zip(episode_numbers, ['A'] + reward_averages))

    with open('n_myopic_n_0_eps_10_time.csv', 'w') as f:
      writer = csv.writer(f, delimiter='\t')
      episode_numbers = ['E'] + range(1, num_episodes + 1)
      writer.writerows(zip(episode_numbers, ['A'] + time_averages))

    with open('n_myopic_n_0_eps_10_res_std_high.csv', 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        episode_numbers = ['E'] + range(1, num_episodes + 1)
        stds_up = map(lambda x: x[0] + x[1], zip(reward_averages, reward_stds))

        writer.writerows(zip(episode_numbers, ['V'] + stds_up))

    with open('n_myopic_n_0_eps_10_res_std_low.csv', 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        episode_numbers = ['E'] + range(1, num_episodes + 1)
        stds_up = map(lambda x: x[0] - x[1], zip(reward_averages, reward_stds))

        writer.writerows(zip(episode_numbers, ['V'] + stds_up))

    with open('n_myopic_n_0_eps_10_time_std_high.csv', 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        episode_numbers = ['E'] + range(1, num_episodes + 1)
        stds_up = map(lambda x: x[0] + x[1], zip(time_averages, time_stds))

        writer.writerows(zip(episode_numbers, ['V'] + stds_up))

    with open('n_myopic_n_0_eps_10_time_std_low.csv', 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        episode_numbers = ['E'] + range(1, num_episodes + 1)
        stds_up = map(lambda x: x[0] - x[1], zip(time_averages, time_stds))

        writer.writerows(zip(episode_numbers, ['V'] + stds_up))



env = gym.make('threat-defense-inspect-v0')
n_myopic(env)
