## Calculate the optimized weight by formula and then output combined predictions and errors

import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB

horizon = 24

def optimize_weights(p, r):
    """
    Minimize (r - weight^T p)^2 subject to weight >= 0, sum(weight) = 1
    """
    num_predictions = p.shape[1]

    model = gp.Model('Weight')
    weight = model.addMVar((num_predictions, 1), lb=0, vtype=GRB.CONTINUOUS)
    model.setObjective(r.T @ r - 2 * r.T @ p @ weight + weight.T @ p.T @ p @ weight, GRB.MINIMIZE)
    model.addConstr(gp.quicksum(weight) == 1)
    model.setParam('OutputFlag', 0) 
    model.optimize()
    return weight.X

def get_mse(real, prediction):
    """
    prediction: periods
    real: periods
    """
    error = prediction - real
    mu_mse = np.mean(np.square(error))
    return mu_mse

def combine_mse(type_data, num_groups, weight):
    """
    Use the obtained weight to generate the combined prediction
    Calculate MSE
    """
    ## Input
    df = pd.read_csv('./data/processed/combination/d031_' + type_data + '.csv')

    ## Use the obtained weight to generate the combined prediction
    df_new = pd.DataFrame({})
    for group in range(num_groups):
        df_new['load' + str(group) + '_real'] = df['load' + str(group) + '_real']
        df_new['load' + str(group) + '_predict'] = np.concatenate((df['load' + str(group) + '_local'].to_numpy().reshape((-1, 1)), df['load' + str(group) + '_HFL'].to_numpy().reshape((-1, 1)), df['load' + str(group) + '_VFL'].to_numpy().reshape((-1, 1))), axis=1) @ weight
    df_new['wind1_real'] = df['wind1_real']
    df_new['wind1_predict'] = df['wind1_predict']
    df_new['wind2_real'] = df['wind2_real']
    df_new['wind2_predict'] = df['wind2_predict']

    df_new.to_csv('./data/processed/combination/d032_' + type_data + '.csv', index=False)

    ## Calculate MSE
    mse_local = np.zeros((num_groups,))
    mse_HFL = np.zeros((num_groups,))
    mse_VFL = np.zeros((num_groups,))
    mse_predict = np.zeros((num_groups,))
    df_mse = pd.DataFrame({})

    for group in range(num_groups):
        mse_local[group] = get_mse(df['load' + str(group) + '_real'], df['load' + str(group) + '_local'])
        mse_HFL[group] = get_mse(df['load' + str(group) + '_real'], df['load' + str(group) + '_HFL'])
        mse_VFL[group] = get_mse(df['load' + str(group) + '_real'], df['load' + str(group) + '_VFL'])
        mse_predict[group] = get_mse(df['load' + str(group) + '_real'], df_new['load' + str(group) + '_predict'])
    df_mse['local'] = mse_local
    df_mse['HFL'] = mse_HFL
    df_mse['VFL'] = mse_VFL
    df_mse['predict'] = mse_predict
    
    df_mse.to_csv('./results/outputs/o032_mse_' + type_data + '.csv', index=False)

def transform(type_data, list_name):
    """
    Transform into matrix form
    """
    df = pd.read_csv('./data/processed/combination/d032_' + type_data + '.csv')
    matrix_real = np.zeros((len(df), len(list_name)))
    matrix_predict = np.zeros((len(df), len(list_name)))
    for i in range(len(list_name)):
        matrix_real[:, i] = df[list_name[i] + '_real']
        matrix_predict[:, i] = df[list_name[i] + '_predict']
    np.save('./data/processed/combination/d032_real_' + type_data + '.npy', matrix_real)
    np.save('./data/processed/combination/d032_predict_' + type_data + '.npy', matrix_predict)

def bounds(list_type):
    """
    Estimate the bounds of errors
    """
    real = np.load('./data/processed/combination/d032_real_' + list_type[0] + '.npy')
    predict = np.load('./data/processed/combination/d032_predict_' + list_type[0] + '.npy')
    error = predict - real
    lb = np.min(error, axis=0)
    ub = np.max(error, axis=0)
    for i in range(1, len(list_type)):
        real = np.load('./data/processed/combination/d032_real_' + list_type[i] + '.npy')
        predict = np.load('./data/processed/combination/d032_predict_' + list_type[i] + '.npy')
        error = predict - real
        lb = np.minimum(lb, np.min(error, axis=0))
        ub = np.maximum(ub, np.max(error, axis=0))
    error_bounds = np.concatenate((lb.reshape((-1, 1)), ub.reshape((-1, 1))), axis=1)
    np.save('./data/processed/uncertainty/d032_error_bounds.npy', error_bounds)

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

## Combine prediction and calculate MSE
combine_mse('train', num_groups, weight)
combine_mse('train_n1', num_groups, weight)
combine_mse('train_n2', num_groups, weight)
combine_mse('validation', num_groups, weight)
combine_mse('test', num_groups, weight)

## Transform dataframes into matrices. Columns are load0-load20, wind1, wind2. Rows are periods.
list_name = ['load' + str(i) for i in range(num_groups)]
list_name.extend(['wind1', 'wind2'])
transform('train', list_name)
transform('train_n1', list_name)
transform('train_n2', list_name)
transform('validation', list_name)
transform('test', list_name)

## Calculate error bounds
list_type = ['train', 'validation', 'test']
bounds(list_type)
