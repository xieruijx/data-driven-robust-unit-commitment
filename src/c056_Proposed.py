## The data-driven RO method using the uncertainty set without reconstruction

import numpy as np
import pandas as pd

from utils.optimization import Optimization

optimization = Optimization()

## Settings
parameter = {}
parameter['type_r'] = 'n1' # type_r: 'n1' max in n1; 'n2' quantile in n2; 'n_m' max in n1 and n2; 'n_q' quantile in n1 and n2
parameter['b_display_SP'] = False
parameter['num_groups'] = 21
parameter['num_wind'] = 4
parameter['horizon'] = 24
parameter['epsilon'] = 0.05 # chance constraint parameter
parameter['delta'] = 0.05 # probability guarantee parameter
parameter['MaxIter'] = 100 # Maximum iteration number of CCG
parameter['LargeNumber'] = 1e8 # For the big-M method
parameter['Tolerance'] = 1e-3 # Tolerance: UB - LB <= Tolerance * UB
parameter['TimeLimitFC'] = 10 # Time limit of the feasibility check problem
parameter['TimeLimitSP'] = 10 # Time limit of the subproblem
parameter['EPS'] = 1e-8 # A small number for margin
parameter['u_select'] = [False, True, True, False, False, False, True,
            False, True, True, True, True, True, True,
            True, False, True, True, True, True, False,
            True, True, False, False] # Only a part of loads and renewables are uncertain

index_u_l_predict = 0

## Set weight as the optimized one
# weight = np.loadtxt('./data/processed/weight/index_9_weight_56.txt')
weight = np.loadtxt('./data/processed/combination/d032_weight.txt')
np.savetxt('./data/processed/weight/index_' + str(index_u_l_predict) + '_weight_Proposed.txt', weight)
validation_cost, test_cost, sxb1, sxc1, LBUB1, time1, sxb2, sxc2, LBUB2, time2, train_cost, train_order, interpret = optimization.weight2cost(parameter, weight, index_u_l_predict)
print(time1)
print(time2)
np.savetxt('./data/processed/weight/index_' + str(index_u_l_predict) + '_validation_cost_Proposed.txt', validation_cost)
np.savetxt('./data/processed/weight/index_' + str(index_u_l_predict) + '_test_cost_Proposed.txt', test_cost)
np.savetxt('./data/processed/weight/index_' + str(index_u_l_predict) + '_LBUB1_Proposed.txt', LBUB1)
np.savetxt('./data/processed/weight/index_' + str(index_u_l_predict) + '_LBUB2_Proposed.txt', LBUB2)
np.savetxt('./data/processed/weight/index_' + str(index_u_l_predict) + '_train_cost_Proposed.txt', train_cost)
np.savetxt('./data/processed/weight/index_' + str(index_u_l_predict) + '_train_order_Proposed.txt', train_order)
np.savetxt('./data/processed/weight/index_' + str(index_u_l_predict) + '_x_og_Proposed.txt', interpret['x_og'])
np.savetxt('./data/processed/weight/index_' + str(index_u_l_predict) + '_x_pg_Proposed.txt', interpret['x_pg'])
np.savetxt('./data/processed/weight/index_' + str(index_u_l_predict) + '_x_rp_Proposed.txt', interpret['x_rp'])
np.savetxt('./data/processed/weight/index_' + str(index_u_l_predict) + '_x_rn_Proposed.txt', interpret['x_rn'])
np.savetxt('./data/processed/weight/index_' + str(index_u_l_predict) + '_y_rp_Proposed.txt', interpret['y_rp'])
np.savetxt('./data/processed/weight/index_' + str(index_u_l_predict) + '_y_rn_Proposed.txt', interpret['y_rn'])
