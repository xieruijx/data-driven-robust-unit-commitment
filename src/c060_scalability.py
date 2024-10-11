import numpy as np

from utils.optimization import Optimization
from utils.case import Case

optimization = Optimization()

## Settings
index_u_l_predict = 0
type_u_l = 'test'

parameter = Case().case118_parameter()
weight_optimize = np.loadtxt('./data/processed/weight118/strategies/n5_weight_test0Proposed.txt')


parameter['u_select'] = [True, True, True, True, True, True, True,
                        True, True, True, True, True, True, True,
                        False, False, False, False, False, False, False,
                        True, True, True, False] # Only a part of loads and renewables are uncertain
validation_cost, test_cost, sxb1, sxc1, LBUB1, sxb2, sxc2, LBUB2, time, train_cost, train_order, interpret = optimization.weight2cost(parameter, weight_optimize, 'n1', None, index_u_l_predict, 'case118', type_u_l)
print(time) # 17

parameter['u_select'] = [True, True, True, True, True, True, True,
                        True, True, True, True, True, True, True,
                        True, True, True, True, True, True, True,
                        True, True, True, True] # Only a part of loads and renewables are uncertain
validation_cost, test_cost, sxb1, sxc1, LBUB1, sxb2, sxc2, LBUB2, time, train_cost, train_order, interpret = optimization.weight2cost(parameter, weight_optimize, 'n1', None, index_u_l_predict, 'case118', type_u_l)
print(time) # 25

parameter['u_select'] = [True, True, True, True, True, True, True,
                        True, False, False, False, False, True, True,
                        True, True, True, True, True, True, True,
                        True, True, True, True] # Only a part of loads and renewables are uncertain
validation_cost, test_cost, sxb1, sxc1, LBUB1, sxb2, sxc2, LBUB2, time, train_cost, train_order, interpret = optimization.weight2cost(parameter, weight_optimize, 'n1', None, index_u_l_predict, 'case118', type_u_l)
print(time) # 21

parameter['u_select'] = [True, True, True, True, True, True, True,
                        True, False, False, False, False, True, True,
                        False, True, False, False, False, False, False,
                        True, False, True, False] # Only a part of loads and renewables are uncertain
validation_cost, test_cost, sxb1, sxc1, LBUB1, sxb2, sxc2, LBUB2, time, train_cost, train_order, interpret = optimization.weight2cost(parameter, weight_optimize, 'n1', None, index_u_l_predict, 'case118', type_u_l)
print(time) # 13