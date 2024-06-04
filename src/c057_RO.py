## The data-driven RO method using the uncertainty set without reconstruction

import numpy as np
import pandas as pd

from utils.optimization import Optimization

optimization = Optimization()

## Settings
parameter = {}
parameter['b_use_n2'] = True # False: the radius is the largest one in n1. True: the radius is the rank one in n2.
parameter['b_display_SP'] = False
parameter['num_groups'] = 21
parameter['horizon'] = 24
parameter['epsilon'] = 0.05 # chance constraint parameter
parameter['delta'] = 0.05 # probability guarantee parameter
parameter['MaxIter'] = 100 # Maximum iteration number of CCG
parameter['LargeNumber'] = 1e8 # For the big-M method
parameter['Tolerance'] = 1e-3 # Tolerance: UB - LB <= Tolerance * UB
parameter['TimeLimitFC'] = 1 # Time limit of the feasibility check problem
parameter['TimeLimitSP'] = 1 # Time limit of the subproblem
parameter['EPS'] = 1e-8 # A small number for margin
parameter['u_select'] = [False, True, True, False, False, False, True,
            False, True, True, True, True, True, True,
            True, False, True, True, True, True, False,
            True, True] # Only a part of loads and renewables are uncertain

index_u_l_predict = 9

## Set weight as the optimized one
weight = np.loadtxt('./data/processed/weight/index_9_weight_56.txt')
np.savetxt('./data/processed/weight/index_' + str(index_u_l_predict) + '_weight_RO.txt', weight)
validation_cost, test_cost, sxb1, sxc1, LBUB1, time1 = optimization.weight2cost_RO(parameter, weight, index_u_l_predict)
cost = np.concatenate((validation_cost.reshape((-1, 1)), test_cost.reshape((-1, 1))), axis=1)
print(time1)
np.savetxt('./data/processed/weight/index_' + str(index_u_l_predict) + '_cost_RO.txt', cost)
np.savetxt('./data/processed/weight/index_' + str(index_u_l_predict) + '_LBUB1_RO.txt', LBUB1)
