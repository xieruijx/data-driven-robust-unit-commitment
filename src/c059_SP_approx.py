## The data-driven SP method using approximate method to solve

import numpy as np

from utils.optimization import Optimization

optimization = Optimization()

## Settings
parameter = {}
parameter['b_faster'] = False # False: MILP; True: Mountain climbing for subproblems in CCG
parameter['b_display_SP'] = False
parameter['num_groups'] = 21
parameter['num_wind'] = 4
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
            True, True, False, False] # Only a part of loads and renewables are uncertain

index_u_l_predict = 0

## Set weight as the optimized one
# weight = np.loadtxt('./data/processed/weight/index_9_weight_56.txt')
weight = np.loadtxt('./data/processed/combination/d032_weight.txt')
np.savetxt('./data/processed/weight/index_' + str(index_u_l_predict) + '_weight_SP_approx.txt', weight)
validation_cost, test_cost, sxb1, sxc1, LBUB1, time1, sxb2, sxc2, LBUB2, time2, train_cost, train_order, interpret = optimization.weight2cost(parameter, weight, 'n1', 'approx', index_u_l_predict, 'case_ieee30')
print(time1)
print(time2)
np.savetxt('./data/processed/weight/index_' + str(index_u_l_predict) + '_validation_cost_SP_approx.txt', validation_cost)
np.savetxt('./data/processed/weight/index_' + str(index_u_l_predict) + '_test_cost_SP_approx.txt', test_cost)
np.savetxt('./data/processed/weight/index_' + str(index_u_l_predict) + '_LBUB1_SP_approx.txt', LBUB1)
np.savetxt('./data/processed/weight/index_' + str(index_u_l_predict) + '_LBUB2_SP_approx.txt', LBUB2)
np.savetxt('./data/processed/weight/index_' + str(index_u_l_predict) + '_train_cost_SP_approx.txt', train_cost)
np.savetxt('./data/processed/weight/index_' + str(index_u_l_predict) + '_train_order_SP_approx.txt', train_order)
np.savetxt('./data/processed/weight/index_' + str(index_u_l_predict) + '_x_og_SP_approx.txt', interpret['x_og'])
np.savetxt('./data/processed/weight/index_' + str(index_u_l_predict) + '_x_pg_SP_approx.txt', interpret['x_pg'])
np.savetxt('./data/processed/weight/index_' + str(index_u_l_predict) + '_x_rp_SP_approx.txt', interpret['x_rp'])
np.savetxt('./data/processed/weight/index_' + str(index_u_l_predict) + '_x_rn_SP_approx.txt', interpret['x_rn'])
np.savetxt('./data/processed/weight/index_' + str(index_u_l_predict) + '_y_rp_SP_approx.txt', interpret['y_rp'])
np.savetxt('./data/processed/weight/index_' + str(index_u_l_predict) + '_y_rn_SP_approx.txt', interpret['y_rn'])

