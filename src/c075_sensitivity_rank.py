## Draw the relationship between rank, epsilon, delta, and number of data
import numpy as np
import matplotlib.pyplot as plt

from utils.combhandler import CombHandler

# Settings
N0 = 124
epsilon0 = 0.05
delta0 = 0.05

set_N = np.arange(1, 3001, 1)
set_rank_N = np.zeros(set_N.shape)
for i in range(set_N.shape[0]):
    set_rank_N[i] = CombHandler().get_rank(set_N[i], epsilon0, delta0)

set_epsilon = np.linspace(0, 0.3, 301)
set_rank_epsilon = np.zeros(set_epsilon.shape)
for i in range(set_epsilon.shape[0]):
    set_rank_epsilon[i] = CombHandler().get_rank(N0, set_epsilon[i], delta0)
# print_epsilon = np.zeros((set_epsilon.shape[0], 2))
# print_epsilon[:, 0] = set_epsilon
# print_epsilon[:, 1] = - set_rank_epsilon + N0
# print(print_epsilon)

set_delta = np.linspace(0, 0.3, 301)
set_rank_delta = np.zeros(set_delta.shape)
for i in range(set_delta.shape[0]):
    set_rank_delta[i] = CombHandler().get_rank(N0, epsilon0, set_delta[i])
# print_delta = np.zeros((set_delta.shape[0], 2))
# print_delta[:, 0] = set_delta
# print_delta[:, 1] = - set_rank_delta + N0
# print(print_delta)

fontsize = 12

fig, ax = plt.subplots(2, 2, figsize=(7, 5))
ax[0, 0].plot(set_N[:500], set_N[:500] - set_rank_N[:500], '-')
ax[0, 0].set_ylabel('Number of points outside', fontsize=fontsize)
ax[0, 0].set_xlabel('$N_2$', fontsize=fontsize)
ax[0, 0].grid(linestyle='--')
ax[0, 1].plot(set_N, (set_N - set_rank_N) / set_N, '-')
ax[0, 1].set_ylabel('Proportion of points outside', fontsize=fontsize)
ax[0, 1].set_xlabel('$N_2$', fontsize=fontsize)
ax[0, 1].grid(linestyle='--')
ax[1, 0].plot(set_epsilon, (- set_rank_epsilon + N0) / N0, '-')
ax[1, 0].set_ylabel('Proportion of points outside', fontsize=fontsize)
ax[1, 0].set_xlabel('$\epsilon$', fontsize=fontsize)
ax[1, 0].grid(linestyle='--')
ax[1, 1].plot(set_delta, (- set_rank_delta + N0) / N0, '-')
ax[1, 1].set_ylabel('Proportion of points outside', fontsize=fontsize)
ax[1, 1].set_xlabel('$\delta$', fontsize=fontsize)
ax[1, 1].grid(linestyle='--')
fig.tight_layout()

plt.show()