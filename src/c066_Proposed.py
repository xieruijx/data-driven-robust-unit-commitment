## The data-driven RO method using the uncertainty set without reconstruction

import numpy as np
import pandas as pd

from utils.optimization import Optimization

optimization = Optimization()

## Settings
name_case = 'case118'

parameter = {}
parameter['b_use_n2'] = False # False: the radius is the largest one in n1. True: the radius is the rank one in n2.
parameter['b_display_SP'] = False
parameter['num_groups'] = 21
parameter['horizon'] = 24
parameter['epsilon'] = 0.05 # chance constraint parameter
parameter['delta'] = 0.05 # probability guarantee parameter
parameter['MaxIter'] = 100 # Maximum iteration number of CCG
parameter['LargeNumber'] = 1e12 # For an initial upper bound
parameter['Tolerance'] = 1e-3 # Tolerance: UB - LB <= Tolerance * UB
parameter['TimeLimitFC'] = 10 # Time limit of the feasibility check problem
parameter['TimeLimitSP'] = 10 # Time limit of the subproblem
parameter['EPS'] = 1e-8 # A small number for margin
parameter['u_select'] = [True, True, True, True, True, True, True,
                        True, True, True, True, True, True, True,
                        True, True, True, True, True, True, True,
                        True, True] # Only a part of loads and renewables are uncertain

index_u_l_predict = 0

## Set weight as the optimized one
weight = np.loadtxt('./data/processed/weight118/index_0_weight_51.txt')
np.savetxt('./data/processed/weight118/index_' + str(index_u_l_predict) + '_weight_Proposed.txt', weight)
validation_cost, test_cost, sxb1, sxc1, LBUB1, time1, sxb2, sxc2, LBUB2, time2, train_cost, train_order = optimization.weight2cost(parameter, weight, index_u_l_predict, name_case)
cost = np.concatenate((validation_cost.reshape((-1, 1)), test_cost.reshape((-1, 1))), axis=1)
print(time1)
print(time2)
np.savetxt('./data/processed/weight118/index_' + str(index_u_l_predict) + '_cost_Proposed.txt', cost)
np.savetxt('./data/processed/weight118/index_' + str(index_u_l_predict) + '_LBUB1_Proposed.txt', LBUB1)
np.savetxt('./data/processed/weight118/index_' + str(index_u_l_predict) + '_LBUB2_Proposed.txt', LBUB2)
np.savetxt('./data/processed/weight118/index_' + str(index_u_l_predict) + '_train_cost_Proposed.txt', train_cost)
np.savetxt('./data/processed/weight118/index_' + str(index_u_l_predict) + '_train_order_Proposed.txt', train_order)
