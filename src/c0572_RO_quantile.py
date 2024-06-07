## The data-driven RO method using the 100% ellipsoidal uncertainty set

import numpy as np

from utils.optimization import Optimization
from utils.case import Case
from utils.io import IO

optimization = Optimization()

## Settings
parameter = Case().case_ieee30_parameter()

index_u_l_predict = 0
type_u_l = 'test'

## Set weight as the optimized one
# weight = np.loadtxt('./data/processed/weight/index_9_weight_56.txt')
weight = np.loadtxt('./data/processed/combination/d032_weight.txt')

validation_cost, test_cost, sxb1, sxc1, LBUB1, sxb2, sxc2, LBUB2, time, train_cost, train_order, interpret = optimization.weight2cost(parameter, weight, 'n_q', None, index_u_l_predict, 'case_ieee30', type_u_l)
print(time)
IO().output_UC(30, index_u_l_predict, type_u_l, weight, train_cost, train_order, validation_cost, test_cost, LBUB1, LBUB2, time, interpret, 'RO_quantile', folder_outputs='./results/outputs/', folder_strategies='./results/strategies/')
