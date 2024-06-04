## Aggregate the cost results of different weights

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import os

## Settings
num_test = 50
epsilon = 0.05
index_u_l_predict = 0
index_cost = 0
np.set_printoptions(precision=5)

rank = math.ceil(num_test * epsilon)

number = pd.read_csv('./data/processed/weight118/number.txt', header=None)
number = number[0]
number = number.iloc[-1] + 1
print(number)

weight_result = np.zeros((number, 3))
cost_result = np.full((number, num_test, 2), np.inf)
cost_derivative = np.full((number, 4, 2), np.inf) # Rank, mean, median, max

for i in range(number):
    weight_result[i, :] = np.loadtxt('./data/processed/weight118/index_' + str(index_u_l_predict) + '_weight_' + str(i) + '.txt')
    if os.path.isfile('./data/processed/weight118/index_' + str(index_u_l_predict) + '_cost_' + str(i) + '.txt'):
        cost_result[i, :, :] = np.loadtxt('./data/processed/weight118/index_' + str(index_u_l_predict) + '_cost_' + str(i) + '.txt')
        for j in range(2):
            cost_derivative[i, 0, j] = cost_result[i, rank - 1, j]
            cost_derivative[i, 1, j] = np.mean(cost_result[i, :, j])
            cost_derivative[i, 2, j] = np.median(cost_result[i, :, j])
            cost_derivative[i, 3, j] = np.max(cost_result[i, :, j])

print(np.concatenate((weight_result, np.arange(0, number).reshape((-1, 1)), cost_derivative[:, 0, :]), axis=1))

X = weight_result[:, :2]
Y = cost_derivative[:, index_cost, 0]
print(Y)
Y[Y == np.inf] = np.max(Y[Y < np.inf]) * 1.01 ##############################################

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
p1 = ax1.scatter(X[:, 0], X[:, 1], Y, c=Y, cmap=plt.cm.rainbow)
fig1.colorbar(p1)
ax1.set_xlabel('weight 1')
ax1.set_ylabel('weight 2')
ax1.set_zlabel('validation cost rank 3')

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
p2 = ax2.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.rainbow)
ax2.plot([0, 0], [0, 1], color='k')
ax2.plot([1, 0], [0, 0], color='k')
ax2.plot([1, 0], [0, 1], color='k')
fig2.colorbar(p2)
ax2.set_xlabel('weight 1')
ax2.set_ylabel('weight 2')

ax2.scatter(X[0, 0], X[0, 1], marker='x', s=150)
ax2.scatter(X[Y == np.min(Y), 0], X[Y == np.min(Y), 1], marker='+', s=150)

weight = [X[Y == np.min(Y), 0], X[Y == np.min(Y), 1], 1 - X[Y == np.min(Y), 0] - X[Y == np.min(Y), 1]]
np.savetxt('./data/processed/combination/d063_weight.txt', weight)

print(np.argsort(Y))

plt.show()