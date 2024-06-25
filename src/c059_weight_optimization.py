## Use the surrogate model to optimize the weight

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from utils.case import Case
from utils.io import IO

## Settings
index_u_l_predict = 16
type_u_l = 'test'
num_points = 50
low = 0
high = 1
num_pca = 3
LargeNumber = 1000
folder_outputs = './data/processed/weight/outputs/'
parameter = Case().case_ieee30_parameter()

X_scaler_mean = np.load(folder_outputs + 'X_scaler_mean.npy')
X_scaler_scale = np.load(folder_outputs + 'X_scaler_scale.npy')
y_scaler_mean = np.load(folder_outputs + 'y_scaler_mean.npy')
y_scaler_scale = np.load(folder_outputs + 'y_scaler_scale.npy')

predict_load, predict_wind = IO().read_test_sample(index_u_l_predict, type_u_l, u_select=parameter['u_select'])
input_pca = IO().to_pca(np.concatenate((predict_load, predict_wind), axis=0).reshape((1, -1)), folder_outputs=folder_outputs).reshape((-1,))

param_model = IO().read_param_model(folder_outputs=folder_outputs)

MILP = gp.Model('Surrogate')
p = {}
q = {}
z = {}
for i in range(param_model['num_layers'] + 1):
    p[i] = MILP.addMVar((param_model['weight' + str(i)].shape[1],), lb=-float('inf'), vtype=GRB.CONTINUOUS)
    q[i] = MILP.addMVar((param_model['weight' + str(i)].shape[0],), lb=-float('inf'), vtype=GRB.CONTINUOUS)
for i in range(param_model['num_layers']):
    z[i] = MILP.addMVar((param_model['weight' + str(i)].shape[0],), vtype=GRB.BINARY)
input = MILP.addMVar(p[0].shape, lb=-float('inf'), vtype=GRB.CONTINUOUS)

print(param_model['num_layers'])
print(q[int(param_model['num_layers'])])
print(q[int(param_model['num_layers'])][0])
MILP.setObjective(q[int(param_model['num_layers'])][0], GRB.MINIMIZE)
for i in range(param_model['num_layers'] + 1):
    MILP.addConstr(q[i] == param_model['weight' + str(i)] @ p[i] + param_model['bias' + str(i)])
for i in range(param_model['num_layers']):
    MILP.addConstr(0 <= p[i + 1])
    MILP.addConstr(p[i + 1] <= LargeNumber * z[i])
    MILP.addConstr(0 <= p[i + 1] - q[i])
    MILP.addConstr(p[i + 1] - q[i] <= LargeNumber * (1 - z[i]))
MILP.addConstr(input[:num_pca] == input_pca[:num_pca])
MILP.addConstr(input[num_pca:] >= 0)
MILP.addConstr(gp.quicksum(input[num_pca:]) <= 1)
MILP.addConstr(p[0] == (input - X_scaler_mean) / X_scaler_scale)
MILP.optimize()
print(input.X)
