## Compare different methods: P1, P2, Proposed, RO_max, RO_quantile, SP_approx
import numpy as np

from utils.optimization import Optimization
from utils.case import Case
from utils.io import IO

optimization = Optimization()

## Settings
index_u_l_predict_set = range(31, 100)
type_u_l = 'test'

parameter = Case().case_ieee30_parameter()
parameter_epsilon0 = Case().case_ieee30_parameter(epsilon=0)

# weight_optimize = np.loadtxt('./data/processed/weight/index_9_weight_56.txt')
weight_optimize = np.loadtxt('./data/processed/combination/d032_weight.txt')
weight_error = np.loadtxt('./data/processed/combination/d032_weight.txt')

for index_u_l_predict in index_u_l_predict_set:
    ## P1: The data-driven RO method using the uncertainty set without reconstruction
    validation_cost, test_cost, sxb1, sxc1, LBUB1, sxb2, sxc2, LBUB2, time, train_cost, train_order, interpret = optimization.weight2cost(parameter, weight_optimize, 'n2', None, index_u_l_predict, 'case_ieee30', type_u_l)
    IO().output_UC(index_u_l_predict, type_u_l, weight_optimize, train_cost, train_order, validation_cost, test_cost, LBUB1, LBUB2, time, interpret, 'P1', folder_outputs='./results/outputs/30/', folder_strategies='./results/strategies/30/')

    ## P2: The data-driven RO method using the weight optimized by minimizing the error measure
    validation_cost, test_cost, sxb1, sxc1, LBUB1, sxb2, sxc2, LBUB2, time, train_cost, train_order, interpret = optimization.weight2cost(parameter, weight_error, 'n1', None, index_u_l_predict, 'case_ieee30', type_u_l)
    IO().output_UC(index_u_l_predict, type_u_l, weight_error, train_cost, train_order, validation_cost, test_cost, LBUB1, LBUB2, time, interpret, 'P2', folder_outputs='./results/outputs/30/', folder_strategies='./results/strategies/30/')

    ## Proposed
    validation_cost, test_cost, sxb1, sxc1, LBUB1, sxb2, sxc2, LBUB2, time, train_cost, train_order, interpret = optimization.weight2cost(parameter, weight_optimize, 'n1', None, index_u_l_predict, 'case_ieee30', type_u_l)
    IO().output_UC(index_u_l_predict, type_u_l, weight_optimize, train_cost, train_order, validation_cost, test_cost, LBUB1, LBUB2, time, interpret, 'Proposed', folder_outputs='./results/outputs/30/', folder_strategies='./results/strategies/30/')

    ## RO_max: The data-driven RO method using the 100% ellipsoidal uncertainty set
    validation_cost, test_cost, sxb1, sxc1, LBUB1, sxb2, sxc2, LBUB2, time, train_cost, train_order, interpret = optimization.weight2cost(parameter, weight_optimize, 'n_m', None, index_u_l_predict, 'case_ieee30', type_u_l)
    IO().output_UC(index_u_l_predict, type_u_l, weight_optimize, train_cost, train_order, validation_cost, test_cost, LBUB1, LBUB2, time, interpret, 'RO_max', folder_outputs='./results/outputs/30/', folder_strategies='./results/strategies/30/')

    ## RO_quantile: The data-driven RO method using the 1 - epsilon ellipsoidal uncertainty set
    validation_cost, test_cost, sxb1, sxc1, LBUB1, sxb2, sxc2, LBUB2, time, train_cost, train_order, interpret = optimization.weight2cost(parameter, weight_optimize, 'n_q', None, index_u_l_predict, 'case_ieee30', type_u_l)
    IO().output_UC(index_u_l_predict, type_u_l, weight_optimize, train_cost, train_order, validation_cost, test_cost, LBUB1, LBUB2, time, interpret, 'RO_quantile', folder_outputs='./results/outputs/30/', folder_strategies='./results/strategies/30/')

    ## SP_approx: The data-driven SP method using approximate method to solve
    validation_cost, test_cost, sxb1, sxc1, LBUB1, sxb2, sxc2, LBUB2, time, train_cost, train_order, interpret = optimization.weight2cost(parameter, weight_optimize, 'n1', 'approx', index_u_l_predict, 'case_ieee30', type_u_l)
    IO().output_UC(index_u_l_predict, type_u_l, weight_optimize, train_cost, train_order, validation_cost, test_cost, LBUB1, LBUB2, time, interpret, 'SP_approx', folder_outputs='./results/outputs/30/', folder_strategies='./results/strategies/30/')

    ## Organize and compare
    IO().organize_methods(index_u_l_predict, type_u_l, 0.05, folder_outputs='./results/outputs/30/')
