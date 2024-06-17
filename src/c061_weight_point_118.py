## Test different weights: Optimize MSE, three vertices
import numpy as np
import pandas as pd

from utils.optimization import Optimization
from utils.case import Case
from utils.io import IO

optimization = Optimization()

## Settings
index_u_l_predict = 0
type_u_l = 'test'
file_number = './data/processed/weight118/number_' + type_u_l + str(index_u_l_predict) + '.txt'

parameter = Case().case118_parameter()

# Weight optimized for minimizing MSE
number = 0
with open(file_number, 'w') as f:
    f.write(str(number) + '\n')
weight = np.loadtxt('./data/processed/combination/d032_weight.txt')
validation_cost, test_cost, sxb1, sxc1, LBUB1, sxb2, sxc2, LBUB2, time, train_cost, train_order, interpret = optimization.weight2cost(parameter, weight, 'n1', None, index_u_l_predict, 'case118', type_u_l)
IO().output_UC(index_u_l_predict, type_u_l, weight, train_cost, train_order, validation_cost, test_cost, LBUB1, LBUB2, time, interpret, 'Proposed', folder_outputs='./data/processed/weight118/outputs/n' + str(number) + '_', folder_strategies='./data/processed/weight118/strategies/n' + str(number) + '_')

# Vertex weights
numbers = pd.read_csv(file_number, header=None)
numbers = numbers[0]
number = numbers.iloc[-1] + 1
with open(file_number, 'w') as f:
    f.write(str(number) + '\n')
weight = np.array([0, 0, 1])
validation_cost, test_cost, sxb1, sxc1, LBUB1, sxb2, sxc2, LBUB2, time, train_cost, train_order, interpret = optimization.weight2cost(parameter, weight, 'n1', None, index_u_l_predict, 'case118', type_u_l)
IO().output_UC(index_u_l_predict, type_u_l, weight, train_cost, train_order, validation_cost, test_cost, LBUB1, LBUB2, time, interpret, 'Proposed', folder_outputs='./data/processed/weight118/outputs/n' + str(number) + '_', folder_strategies='./data/processed/weight118/strategies/n' + str(number) + '_')

numbers = pd.read_csv(file_number, header=None)
numbers = numbers[0]
number = numbers.iloc[-1] + 1
with open(file_number, 'w') as f:
    f.write(str(number) + '\n')
weight = np.array([0, 1, 0])
validation_cost, test_cost, sxb1, sxc1, LBUB1, sxb2, sxc2, LBUB2, time, train_cost, train_order, interpret = optimization.weight2cost(parameter, weight, 'n1', None, index_u_l_predict, 'case118', type_u_l)
IO().output_UC(index_u_l_predict, type_u_l, weight, train_cost, train_order, validation_cost, test_cost, LBUB1, LBUB2, time, interpret, 'Proposed', folder_outputs='./data/processed/weight118/outputs/n' + str(number) + '_', folder_strategies='./data/processed/weight118/strategies/n' + str(number) + '_')

numbers = pd.read_csv(file_number, header=None)
numbers = numbers[0]
number = numbers.iloc[-1] + 1
with open(file_number, 'w') as f:
    f.write(str(number) + '\n')
weight = np.array([1, 0, 0])
validation_cost, test_cost, sxb1, sxc1, LBUB1, sxb2, sxc2, LBUB2, time, train_cost, train_order, interpret = optimization.weight2cost(parameter, weight, 'n1', None, index_u_l_predict, 'case118', type_u_l)
IO().output_UC(index_u_l_predict, type_u_l, weight, train_cost, train_order, validation_cost, test_cost, LBUB1, LBUB2, time, interpret, 'Proposed', folder_outputs='./data/processed/weight118/outputs/n' + str(number) + '_', folder_strategies='./data/processed/weight118/strategies/n' + str(number) + '_')
