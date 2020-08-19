import numpy as np


def softmax_convert_into_pi_from_theta(theta):
    beta = 1.0
    [m, n] = theta.shape
    pi = np.zeros((m, n))

    exp_theta = np.exp(beta * theta)

    for i in range(m):
        pi[i, :] = exp_theta[i, :] / np.nansum(exp_theta[i, :])

    pi = np.nan_to_num(pi)

    return pi


pi_0 = softmax_convert_into_pi_from_theta(theta_0)
print(pi_0)


def get_action_and_next_s(pi, s):
    direction = ["up", "right", "down", "left"]
    next_direction = np.random.choice(direction, p=pi[s, :])

    if next_direction == "up":
        action = 0
        s_next = s - 3
    elif next_direction == "right":
        action = 1
        s_next = s + 1
    elif next_direction == "down":
        action = 2
        s_next = s + 3
    elif next_direction == "left":
        action = 3
        s_next = s - 1

    return [action, s_next]


def goal_maze_ret_s_a(pi):
    s = 0
    s_a_history = [[0, np.nan]]

    while True:
        [action, next_s] = get_action_and_next_s(pi, s)
        s_a_history[-1][1] = action
        s_a_history.append([next_s, np.nan])

        if next_s == 8:
            break
        else:
            s = next_s
    return s_a_history


s_a_history = goal_mze_ret_s_a(pi_0)
print(s_a_history)


def update_theta(theta, pi, s_a_history):
    eta = 0.1
    T = len(s_a_history) - 1
    [m, n] = theta.shape
    delta_theta = theta.copy()

    for i in range(m):
        for j in range(n):
            SA_i = [SA for SA in s_a_history if SA[0] == i]
            SA_ij = [SA for SA in s_a_history if SA == [i, j]]
            N_i = len(SA_i)
            N_ij = len(SA_ij)
            delta_theta[i, j] = (N_ij - pi[i, j] * N_i) / T
    new_theta = theta + eta * delta_theta

    return new_theta


new_theta = update_theta(theta_0, pi_0, s_a_history)
pi = softmax_convert_into_pi_from_theta(new_theta)
print(pi)

stop_epsilon = 10 ** -4
theta = theta_0
pi = pi_0

is_continue = True
count = 1
while is_continue:
    s_a_history = goal_maze_ret_s_a(pi)
    new_theta = update_theta(theta, pi, s_a_history)
    new_pi = softmax_convert_into_pi_from_theta(new_theta)
    print(np.sum(np.abs(new_pi - pi)))

    if np.sum(np.abs(new_pi - pi)) < stop_epsilon:
        is_continue = False
    else:
        theta = new_theta
        pi = new_pi

np.set_printoptions(precision=3, suppress=True)
print(pi)


def simple_convert_into_pi_from_theta(theta):
    [m, n] = theta.shape
    pi = np.zeros((m, n))
    for i in range(m):
        pi[i, :] = theta[i, :] / np.nansum(theta[i, :])
    pi = np.nan_to_num(pi)
    return pi


pi_0 = simple_convert_into_pi_from_theta(theta_0)


def get_actions(s, Q, epsilon, pi_0):
    direction = ["up", "right", "down", "left"]

    if np.random.rand() < epsilon:
        next_direction = np.random.choice(direction, p=pi_0[s, :])
    else:
        next_direction = direction[np.nanargmax(Q[s, :])]
    if next_direction == "up":
        action = 0
    elif next_direction == "right":
        action = 1
    elif next_direction == "down":
        action = 2
    elif next_direction == "left":
        action = 3
    return action

def get_s_next(s,a,Q,epsilon, pi_0):
    direction = ["up", "right", "down", "left"]
    next_direction = direction[a]
    if next_direction == "up":
        s_next = s - 3
    elif next_direction == "right":
        s_next = s + 1
    elif next_direction == "down":
        s_next = s + 3
    elif next_direction == "left":
        s_next = s - 1
    return s_next

def Sarsa(s,a,r,s_next,a_next,Q,eta,gamma):
    if s_next == 8:
        Q[s,a] += eta * (r-Q[s,a])
    else:
        Q[s,a] += eta * (r + gamma * Q[s_next,a_next]-Q[s,a])
    return Q

def goal_maze_ret_s_a_Q(Q,epsilon,eta,gamma,pi):
    s = 0
    a = a_next = get_actions(s,Q,epsilon,pi)
    s_a_history = [[0,np.nan]]

    while True:
        a = a_next
        s_a_history[-1][1] = a
        s_next = get_s_next(s,a,Q,epsilon,pi)
        s_a_history.append([s_next,np.nan])

        if s_next == 8:
            r = 1
            a_next = np.nan
        else:
            r = 0
            a_next = get_actions(s_next, Q, epsilon, pi)
        Q = Q_learning(s,a,r,s_next,Q,eta,gamma)

        if s_next == 8:
            break
        else:
            s = s_next
    return [s_a_history, Q]

eta = 0.1
gamma = 0.9
epsilon = 0.5
v = np.nanmax(Q, axis = 1)
is_continue = True
episode = 1

while is_continue:
    print(episode)
    epsilon = episode/2

    [s_a_history, Q] = goal_maze_ret_s_a_Q(Q,epsilon,eta,gamma,pi_0)

    new_v = np.nanmax(Q, axis=1)
    print(np.sum(np.abs(new_v-v)))
    v = new_v
    print(len(s_a_history)-1)
    episode += 1
    if episode > 100:
        break

def Q_learning(s,a,r,s_next, Q,eta,gamma):
    if s_next == 8:
        Q[s,a] += eta * (r - Q[s,a])
    else:
        Q[s,a] += eta* (r + gamma * np.nanmax(Q[s_next, :]) - Q[s,a])
    return Q

[a,b] = theta_0.shape

Q = np.random.rand(a,b) * theta_0 * 0.1

eta = 0.1
gamma = 0.9
epsilon = 0.5
v = np.nanmax(Q, axis=1)
is_continue = True
episode = 1
V = []
while is_continue:
    print(episode)
    epsilon /= 2
    [s_a_history, Q] = goal_maze_ret_s_a_Q(Q, epsilon, eta, gamma, pi_0)
    new_v = np.nanmax(Q, axis=1)
    print(np.sum(np.abs(new_v-v)))
    v = new_v
    V.append(v)
    print(len(s_a_history)-1)
    episode += 1
    if episode > 100:
        break

def goal_maze_ret_s_a_Q(Q,epsilon,eta,gamma,pi):
    s = 0
    a = a_next = get_actions(s,Q,epsilon,pi)
    s_a_history = [[0,np.nan]]

    while True:
        a = a_next
        s_a_history[-1][1] = a
        s_next = get_s_next(s,a,Q,epsilon,pi)
        s_a_history.append([s_next,np.nan])

        if s_next == 8:
            r = 1
            a_next = np.nan
        else:
            r = 0
            a_next = get_actions(s_next, Q, epsilon, pi)
        Q = Q_learning(s,a,r,s_next,Q,eta,gamma)

        if s_next == 8:
            break
        else:
            s = s_next
    return [s_a_history, Q]
