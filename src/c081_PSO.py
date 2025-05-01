## Using PSO to optimize the weight
import numpy as np
import sys
import os
import time

from utils.optimization import Optimization
from utils.case import Case
from utils.PSO import Particle, pso
from utils.io import IO

optimization = Optimization()

## Settings
index_u_l_predict = 16
type_u_l = 'test'
epsilon = 0.05

parameter = Case().case_ieee30_parameter()

def sphere_function(x):
    return sum((x-0.5)**2)

def loss_function(x):
    if x[0] + x[1] <= 1:
        weight = np.array([x[0], x[1], 1 - x[0] - x[1]])
    else:
        weight = np.array([x[0] / (x[0] + x[1]), x[1] / (x[0] + x[1]), 0])

    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        try:
            validation_cost, test_cost, sxb1, sxc1, LBUB1, sxb2, sxc2, LBUB2, time, train_cost, train_order, interpret = optimization.weight2cost(parameter, weight, 'n1', None, index_u_l_predict, 'case_ieee30', type_u_l)
            cost = validation_cost[np.argsort(validation_cost)[np.ceil((1 - epsilon) * validation_cost.shape[0]).astype(int) - 1]]
        except Exception as e:
            print(f"Error: {e}")
            cost = 1e8
    finally:
        sys.stdout = original_stdout

    print(f"Weight: {weight[0]}, {weight[1]}, {weight[2]}")
    print(f"Cost: {cost}")

    return cost

# start_time = time.time()

# best_position, best_value = pso(loss_function)

# end_time = time.time()
# print(f"Solution time: {end_time - start_time:.4f} s")

# print("Best Position:", best_position)
# print("Best Value:", best_value)

best_position = np.array([0, 0.52067, 1-0.52067])

validation_cost, test_cost, sxb1, sxc1, LBUB1, sxb2, sxc2, LBUB2, time, train_cost, train_order, interpret = optimization.weight2cost(parameter, best_position, 'n1', None, index_u_l_predict, 'case_ieee30', type_u_l)
IO().output_UC(index_u_l_predict, type_u_l, best_position, train_cost, train_order, validation_cost, test_cost, LBUB1, LBUB2, time, interpret, 'PSO', folder_outputs='./results/outputs/30/', folder_strategies='./results/strategies/30/')
cost = test_cost[index_u_l_predict]
print(f"Test Cost: {cost}")
