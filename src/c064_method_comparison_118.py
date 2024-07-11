## Compare different methods: P1, P2, Proposed, RO_max, RO_quantile, SP_approx
import numpy as np

from utils.optimization import Optimization
from utils.case import Case
from utils.io import IO

optimization = Optimization()

## Settings
index_u_l_predict_set = [0]
type_u_l = 'test'

parameter = Case().case118_parameter()
parameter_epsilon0 = Case().case118_parameter(epsilon=0)
parameter_smallMaxIter = Case().case118_parameter(MaxIter=20)

weight_optimize = np.loadtxt('./data/processed/weight118/strategies/n5_weight_test0Proposed.txt') # 10
weight_error = np.loadtxt('./data/processed/combination/d032_weight.txt')

for index_u_l_predict in index_u_l_predict_set:
    ## P2: The data-driven RO method using the weight optimized by minimizing the error measure
    validation_cost, test_cost, sxb1, sxc1, LBUB1, sxb2, sxc2, LBUB2, time, train_cost, train_order, interpret = optimization.weight2cost(parameter, weight_error, 'n1', None, index_u_l_predict, 'case118', type_u_l)
    IO().output_UC(index_u_l_predict, type_u_l, weight_error, train_cost, train_order, validation_cost, test_cost, LBUB1, LBUB2, time, interpret, 'P2', folder_outputs='./results/outputs/118/', folder_strategies='./results/strategies/118/')

    ## RO_max: The data-driven RO method using the 100% ellipsoidal uncertainty set
    validation_cost, test_cost, sxb1, sxc1, LBUB1, sxb2, sxc2, LBUB2, time, train_cost, train_order, interpret = optimization.weight2cost(parameter, weight_error, 'n_m', None, index_u_l_predict, 'case118', type_u_l)
    IO().output_UC(index_u_l_predict, type_u_l, weight_error, train_cost, train_order, validation_cost, test_cost, LBUB1, LBUB2, time, interpret, 'RO_max', folder_outputs='./results/outputs/118/', folder_strategies='./results/strategies/118/')
    
    ## P1: The data-driven RO method using the uncertainty set without reconstruction
    validation_cost, test_cost, sxb1, sxc1, LBUB1, sxb2, sxc2, LBUB2, time, train_cost, train_order, interpret = optimization.weight2cost(parameter, weight_optimize, 'n2', None, index_u_l_predict, 'case118', type_u_l)
    IO().output_UC(index_u_l_predict, type_u_l, weight_optimize, train_cost, train_order, validation_cost, test_cost, LBUB1, LBUB2, time, interpret, 'P1', folder_outputs='./results/outputs/118/', folder_strategies='./results/strategies/118/')

    ## Proposed
    validation_cost, test_cost, sxb1, sxc1, LBUB1, sxb2, sxc2, LBUB2, time, train_cost, train_order, interpret = optimization.weight2cost(parameter, weight_optimize, 'n1', None, index_u_l_predict, 'case118', type_u_l)
    IO().output_UC(index_u_l_predict, type_u_l, weight_optimize, train_cost, train_order, validation_cost, test_cost, LBUB1, LBUB2, time, interpret, 'Proposed', folder_outputs='./results/outputs/118/', folder_strategies='./results/strategies/118/')

    ## SP_approx: The data-driven SP method using approximate method to solve
    validation_cost, test_cost, sxb1, sxc1, LBUB1, sxb2, sxc2, LBUB2, time, train_cost, train_order, interpret = optimization.weight2cost(parameter_smallMaxIter, weight_error, 'n1', 'approx', index_u_l_predict, 'case118', type_u_l)
    IO().output_UC(index_u_l_predict, type_u_l, weight_error, train_cost, train_order, validation_cost, test_cost, LBUB1, LBUB2, time, interpret, 'SP_approx', folder_outputs='./results/outputs/118/', folder_strategies='./results/strategies/118/')

    ## Organize and compare
    IO().organize_methods(index_u_l_predict, type_u_l, 0.05, folder_outputs='./results/outputs/118/')
