## Organize training data for the surrogate model of optimizing weight

from utils.case import Case
from utils.io import IO

type_u_l = 'test'
parameter = Case().case_ieee30_parameter()

matrix_load, matrix_wind, matrix_weight, matrix_cost = IO().read_training_data(type_u_l, u_select=parameter['u_select'], name_method='Proposed')
print(matrix_load.shape)
print(matrix_wind.shape)
print(matrix_weight.shape)
print(matrix_cost.shape)
