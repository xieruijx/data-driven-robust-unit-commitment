## The data-driven RO method using the uncertainty set without reconstruction

import numpy as np
import pandas as pd

from utils.optimization import Optimization

optimization = Optimization()

## Settings
parameter = {}
parameter['type_r'] = None # type_r: 'n1' max in n1; 'n2' quantile in n2; 'n_m' max in n1 and n2; 'n_q' quantile in n1 and n2
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
np.savetxt('./data/processed/weight/index_' + str(index_u_l_predict) + '_weight_SP.txt', weight)
validation_cost, test_cost, sxb1, sxc1, LBUB1, time1 = optimization.weight2cost_SPapprox(parameter, weight, index_u_l_predict)
cost = np.concatenate((validation_cost.reshape((-1, 1)), test_cost.reshape((-1, 1))), axis=1)
print(time1)
np.savetxt('./data/processed/weight/index_' + str(index_u_l_predict) + '_cost_SP.txt', cost)
np.savetxt('./data/processed/weight/index_' + str(index_u_l_predict) + '_LBUB1_SP.txt', LBUB1)
