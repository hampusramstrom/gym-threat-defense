

import numpy as np
import gym
import gym_threat_defense  # noqa


def choose_action(env, i, o_list, r, a, emp_inspect, opt_a, eps, trans_matrix, n):  # noqa
    if i < eps:
        if (a == 1 and
           get_index_in_matrix(env, o_list[1]) == env.state_space.n - 2):
            return np.random.randint(2, env.action_space.n - 1)
        else:
            return 1
    else:
        ind_o = get_index_in_matrix(env, o_list[0])

        prob = 1 - emp_inspect[ind_o, env.state_space.n - 2]
        state = emp_inspect[ind_o]
        tmp = np.zeros(env.state_space.n)

        for _ in range(n):
            for i in range(trans_matrix.shape[1]):
                tmp[i] = state.dot(trans_matrix[:, i])
            prob = prob * (1 - tmp[env.state_space.n - 2])
            state = tmp
            tmp = np.zeros(env.state_space.n)

        tot_prob = 1 - prob

        if tot_prob > 0.5 or ind_o == env.state_space.n - 1:
            return opt_a
        else:
            return 0


def get_index_in_matrix(env, observation):
    for i in range(env.all_states.shape[0]):
        if np.array_equal(observation, env.all_states[i]):
            return i


def update_matrixes(emp_inspect, trans_matrix, a, o_list, old_ind_s):
    if a == 1:
        ind_o = get_index_in_matrix(env, o_list[0])
        ind_s = get_index_in_matrix(env, o_list[1])
        emp_inspect[ind_o, ind_s] += 1
        if old_ind_s == env.state_space.n - 2:
            trans_matrix[old_ind_s, env.state_space.n - 1] += 1
        else:
            trans_matrix[old_ind_s, ind_s] += 1
        old_ind_s = ind_s
    return emp_inspect, trans_matrix, old_ind_s


def n_lookahead(env, n): # noqa: C901
    emp_inspect = np.zeros([env.observation_space.n, env.state_space.n])
    trans_matrix = np.zeros([env.state_space.n, env.state_space.n])

    emp_inspect[env.observation_space.n - 1, env.state_space.n - 1] = 1
    trans_matrix[env.observation_space.n - 1, env.state_space.n - 1] = 1

    num_episodes = 1000
    eps = 100
    rewards = []

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
        old_ind_s = 0

        t_since_a = 0
        last_a = env.action_space.n

        while True:
            a = choose_action(env, i, o_list, old_r, old_a, emp_inspect,
                              opt_a, eps, trans_matrix, n)
            o_list, r, done, info = env.step(a)

            emp_inspect, trans_matrix, old_ind_s = update_matrixes(
                emp_inspect, trans_matrix, a, o_list, old_ind_s)

            r_all += r
            old_r = r
            old_a = a

            t_since_a += 1

            if i < eps and a > 1:
                if last_a < env.action_space.n:
                    action_steps_matrix[
                        last_a - 2,
                        action_steps_ind[last_a - 2]] = t_since_a
                    action_steps_ind[last_a - 2] += 1
                t_since_a = 0
                last_a = a
            elif i == eps:
                for j in range(len(action_steps_matrix)):
                    if np.count_nonzero(action_steps_matrix[j]) == 0:
                        average_len_of_a[j] = \
                            np.sum(action_steps_matrix[j]) / \
                            abs(action_costs[j + 2])
                    else:
                        average_len_of_a[j] = (
                            np.sum(action_steps_matrix[j]) /
                            np.count_nonzero(action_steps_matrix[j])) / \
                            abs(action_costs[j + 2])
                opt_a = np.argmax(average_len_of_a) + 2

                for j in range(env.state_space.n):
                    if np.sum(emp_inspect[j]) == 0 and \
                                              np.sum(trans_matrix[j]) == 0:
                        emp_inspect[j] = emp_inspect[j] / 1
                        trans_matrix[j] = trans_matrix[j] / 1
                    elif np.sum(emp_inspect[j]) == 0:
                        emp_inspect[j] = emp_inspect[j] / 1
                        trans_matrix[j] = trans_matrix[j] / \
                            np.sum(trans_matrix[j])
                    elif np.sum(trans_matrix[j]) == 0:
                        emp_inspect[j] = emp_inspect[j] / \
                            np.sum(emp_inspect[j])
                        trans_matrix[j] = trans_matrix[j] / 1
                    else:
                        emp_inspect[j] = emp_inspect[j] / \
                            np.sum(emp_inspect[j])
                        trans_matrix[j] = trans_matrix[j] / \
                            np.sum(trans_matrix[j])

            if done:
                break

        rewards.append(r_all)
        if i % 100 == 0 and i > 0:
            print "Episode: %s" % i
            print "Score over last 100 episodes: " + \
                str(sum(rewards[(i - 100):i]) / 100)

    np.set_printoptions(suppress=True)


env = gym.make('threat-defense-inspect-v0')
n_lookahead(env, 0)
