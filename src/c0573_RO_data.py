## The data-driven SP method using approximate method to solve

import numpy as np

from utils.optimization import Optimization
from utils.case import Case

optimization = Optimization()

## Settings
parameter = Case().case_ieee30_parameter(epsilon=0)

index_u_l_predict = 0

## Set weight as the optimized one
# weight = np.loadtxt('./data/processed/weight/index_9_weight_56.txt')
weight = np.loadtxt('./data/processed/combination/d032_weight.txt')
np.savetxt('./data/processed/weight/index_' + str(index_u_l_predict) + '_weight_RO_data.txt', weight)
validation_cost, test_cost, sxb1, sxc1, LBUB1, time1, sxb2, sxc2, LBUB2, time2, train_cost, train_order, interpret = optimization.weight2cost(parameter, weight, 'n1', 'approx', index_u_l_predict, 'case_ieee30')
print(time1)
print(time2)
np.savetxt('./data/processed/weight/index_' + str(index_u_l_predict) + '_validation_cost_RO_data.txt', validation_cost)
np.savetxt('./data/processed/weight/index_' + str(index_u_l_predict) + '_test_cost_RO_data.txt', test_cost)
np.savetxt('./data/processed/weight/index_' + str(index_u_l_predict) + '_LBUB1_RO_data.txt', LBUB1)
np.savetxt('./data/processed/weight/index_' + str(index_u_l_predict) + '_LBUB2_RO_data.txt', LBUB2)
np.savetxt('./data/processed/weight/index_' + str(index_u_l_predict) + '_train_cost_RO_data.txt', train_cost)
np.savetxt('./data/processed/weight/index_' + str(index_u_l_predict) + '_train_order_RO_data.txt', train_order)
np.savetxt('./data/processed/weight/index_' + str(index_u_l_predict) + '_x_og_RO_data.txt', interpret['x_og'])
np.savetxt('./data/processed/weight/index_' + str(index_u_l_predict) + '_x_pg_RO_data.txt', interpret['x_pg'])
np.savetxt('./data/processed/weight/index_' + str(index_u_l_predict) + '_x_rp_RO_data.txt', interpret['x_rp'])
np.savetxt('./data/processed/weight/index_' + str(index_u_l_predict) + '_x_rn_RO_data.txt', interpret['x_rn'])
np.savetxt('./data/processed/weight/index_' + str(index_u_l_predict) + '_y_rp_RO_data.txt', interpret['y_rp'])
np.savetxt('./data/processed/weight/index_' + str(index_u_l_predict) + '_y_rn_RO_data.txt', interpret['y_rn'])
