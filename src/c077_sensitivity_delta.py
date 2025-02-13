## Compare different epsilon
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.optimization import Optimization
from utils.case import Case
from utils.io import IO

optimization = Optimization()

## Settings
index_u_l_predict = 16
type_u_l = 'test'
set_delta = np.array([0, 0.050, 0.128, 0.253])
num_delta = set_delta.shape[0]
TimeLimit = 1
weight_optimize = np.loadtxt('./data/processed/combination/d053_weight.txt')
folder_outputs = './results/outputs/30/'
folder_strategies = './results/strategies/30/'
epsilon = 0.05

set_folder_outputs = {}
set_folder_strategies = {}
for i in range(num_delta):
    set_folder_outputs[i] = folder_outputs + 'dn' + str(i) + '/'
    if not os.path.isdir(set_folder_outputs[i]):
        os.mkdir(set_folder_outputs[i])
    set_folder_strategies[i] = folder_strategies + 'dn' + str(i) + '/'
    if not os.path.isdir(set_folder_strategies[i]):
        os.mkdir(set_folder_strategies[i])

# ## Computation
# for i in range(num_delta):
#     parameter_delta = Case().case_ieee30_parameter(delta=set_delta[i], TimeLimit=TimeLimit)

#     validation_cost, test_cost, sxb1, sxc1, LBUB1, sxb2, sxc2, LBUB2, time, train_cost, train_order, interpret = optimization.weight2cost(parameter_delta, weight_optimize, 'n1', None, index_u_l_predict, 'case_ieee30', type_u_l)

#     IO().output_UC(index_u_l_predict, type_u_l, weight_optimize, train_cost, train_order, validation_cost, test_cost, LBUB1, LBUB2, time, interpret, 'Proposed', folder_outputs=set_folder_outputs[i], folder_strategies=set_folder_strategies[i])

## Organize
outputs = {}
for i in range(num_delta):
    train_cost = np.loadtxt(set_folder_outputs[i] + 'train_cost_' + type_u_l + str(index_u_l_predict) + 'Proposed.txt')
    train_q = train_cost[np.argsort(train_cost)[np.ceil((1 - epsilon) * train_cost.shape[0]).astype(int) - 1]]
    train_r = - np.sum(np.isinf(train_cost)) / train_cost.shape[0] + 1

    validation_cost = np.loadtxt(set_folder_outputs[i] + 'validation_cost_' + type_u_l + str(index_u_l_predict) + 'Proposed.txt')
    validation_q = validation_cost[np.argsort(validation_cost)[np.ceil((1 - epsilon) * validation_cost.shape[0]).astype(int) - 1]]
    validation_r = - np.sum(np.isinf(validation_cost)) / validation_cost.shape[0] + 1

    test_cost = np.loadtxt(set_folder_outputs[i] + 'test_cost_' + type_u_l + str(index_u_l_predict) + 'Proposed.txt')
    test_q = test_cost[np.argsort(test_cost)[np.ceil((1 - epsilon) * test_cost.shape[0]).astype(int) - 1]]
    test_r = - np.sum(np.isinf(test_cost)) / test_cost.shape[0] + 1

    cost = test_cost[index_u_l_predict]

    LBUB2 = np.loadtxt(set_folder_outputs[i] + 'LBUB2_' + type_u_l + str(index_u_l_predict) + 'Proposed.txt', ndmin=2)
    obj = LBUB2[-1, 0]

    outputs[i] = [set_delta[i], train_q, train_r, validation_q, validation_r, test_q, test_r, cost, obj]

df = pd.DataFrame(outputs, index=['delta', 'train quantile', 'train rate', 'validation quantile', 'validation rate', 'test quantile', 'test rate', 'cost', 'objective']).T
print(df)
print(df[['delta', 'test quantile', 'test rate', 'cost', 'objective']])

## Draw
fontsize = 12

fig, ax = plt.subplots(1, 2, figsize=(7, 2.2))
ax[0].plot(df['delta'], df['objective']+69, 'o-', label='Objective')
ax[0].set_ylabel('Objective (\$)', fontsize=fontsize)
ax[0].set_xlabel('$\delta$', fontsize=fontsize)
ax[0].grid(linestyle='--')
ax[0].set_ylim((89000, 90000))
ax[1].plot(df['delta'], df['test rate'], 'o-')
ax[1].set_xlabel('$\delta$', fontsize=fontsize)
ax[1].set_ylabel('Feasible rate', fontsize=fontsize)
ax[1].grid(linestyle='--')
ax[1].set_ylim((0.85, 1))

plt.tight_layout()
plt.show()
