## Calculate the optimized weight by formula and then output combined predictions and errors

import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB

horizon = 24

def optimize_weights(p, r):
    """
    Minimize (r - weight^T p)^2 subject to sum(weight) = 1
    """
    num_predictions = p.shape[1]

    model = gp.Model('Weight')
    weight = model.addMVar((num_predictions, 1), lb=-float('inf'), vtype=GRB.CONTINUOUS)
    model.setObjective(r.T @ r - 2 * r.T @ p @ weight + weight.T @ p.T @ p @ weight, GRB.MINIMIZE)
    model.addConstr(gp.quicksum(weight) == 1)
    model.setParam('OutputFlag', 0) 
    model.optimize()
    return weight.X

## Input
df_validation = pd.read_csv('./data/processed/combination/d031_validation.csv')

## Data processing
num_groups = 21
p_validation = np.concatenate((df_validation['load0_local'].to_numpy().reshape((-1, 1)), df_validation['load0_HFL'].to_numpy().reshape((-1, 1)), df_validation['load0_VFL'].to_numpy().reshape((-1, 1))), axis=1)
r_validation = df_validation['load0_real'].to_numpy().reshape((-1, 1))
for group in range(1, num_groups):
    p_validation = np.concatenate((p_validation, np.concatenate((df_validation['load' + str(group) + '_local'].to_numpy().reshape((-1, 1)), df_validation['load' + str(group) + '_HFL'].to_numpy().reshape((-1, 1)), df_validation['load' + str(group) + '_VFL'].to_numpy().reshape((-1, 1))), axis=1)), axis=0)
    r_validation = np.concatenate((r_validation, df_validation['load' + str(group) + '_real'].to_numpy().reshape((-1, 1))), axis=0)

## Use the common weight that is optimal among the validation data
weight = optimize_weights(p_validation, r_validation)
np.savetxt('./data/processed/combination/d032_weight.txt', weight)
