import numpy as np

from utils.optimization import Optimization
from utils.case import Case

optimization = Optimization()

## Settings
index_u_l_predict = 0
type_u_l = 'test'

parameter = Case().case_ieee30_parameter()
weight_optimize = np.loadtxt('./data/processed/combination/d053_weight.txt')

parameter['u_select'] = [False, True, True, False, False, False, True,
                    False, True, True, True, True, True, True,
                    True, False, True, True, True, True, False,
                    True, True, False, False] # Only a part of loads and renewables are uncertain
validation_cost, test_cost, sxb1, sxc1, LBUB1, sxb2, sxc2, LBUB2, time16, train_cost, train_order, interpret = optimization.weight2cost(parameter, weight_optimize, 'n1', None, index_u_l_predict, 'case_ieee30', type_u_l)
print(time16)

parameter['u_select'] = [False, True, True, False, False, False, True,
                    False, True, True, True, True, True, True,
                    True, False, True, True, True, False, False,
                    False, False, False, False] # Only a part of loads and renewables are uncertain
validation_cost, test_cost, sxb1, sxc1, LBUB1, sxb2, sxc2, LBUB2, time13, train_cost, train_order, interpret = optimization.weight2cost(parameter, weight_optimize, 'n1', None, index_u_l_predict, 'case_ieee30', type_u_l)
print(time13)

parameter['u_select'] = [False, True, True, False, False, False, True,
                    False, True, True, True, True, True, True,
                    True, False, False, False, False, False, False,
                    False, False, False, False] # Only a part of loads and renewables are uncertain
validation_cost, test_cost, sxb1, sxc1, LBUB1, sxb2, sxc2, LBUB2, time10, train_cost, train_order, interpret = optimization.weight2cost(parameter, weight_optimize, 'n1', None, index_u_l_predict, 'case_ieee30', type_u_l)
print(time10)

parameter['u_select'] = [False, True, True, False, False, False, True,
                    False, True, True, True, True, False, False,
                    False, False, False, False, False, False, False,
                    False, False, False, False] # Only a part of loads and renewables are uncertain
validation_cost, test_cost, sxb1, sxc1, LBUB1, sxb2, sxc2, LBUB2, time7, train_cost, train_order, interpret = optimization.weight2cost(parameter, weight_optimize, 'n1', None, index_u_l_predict, 'case_ieee30', type_u_l)
print(time7)
