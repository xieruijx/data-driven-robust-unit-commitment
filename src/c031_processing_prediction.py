## Combine load and renewable data. Divide into train, validation, and test data sets. Divide train into n1 and n2.

import pandas as pd
import numpy as np
import random

random.seed(0)

horizon = 24
sequence_length = 192
num_rows = 12864
num_groups = 21
num_wind = 4
n1 = 212
n_validation = 100
n_test = 100
num_data = num_rows // horizon
input_day = sequence_length // horizon - 1
n2 = num_data - input_day - n1 - n_validation - n_test

## Labels for deviding data
day_status = np.zeros((num_data,))
# 1: train
# 2: validation !!! not the same as that in forecasting
# 3: test !!! not the same as that in forecasting
# 0: others in total
order = list(range(num_data - input_day))
random.shuffle(order)
np.savetxt('./data/processed/combination/d031_order_n_validation_test.txt', np.array(order), fmt='%d')
index_train = [input_day + x for x in order[:(n1 + n2)]]
day_status[index_train] = 1
index_validation = [input_day + x for x in order[(n1 + n2):(n1 + n2 + n_validation)]]
day_status[index_validation] = 2
index_test = [input_day + x for x in order[(n1 + n2 + n_validation):]]
day_status[index_test] = 3

period_status = np.zeros((num_rows,))
for i in range(num_data):
    period_status[(i * horizon):((i + 1) * horizon)] = day_status[i]

## Initiation
df_train = pd.DataFrame({})
df_validation = pd.DataFrame({})
df_test = pd.DataFrame({})

## Loads
for group in range(num_groups):
    df_local = pd.read_csv('./data/processed/load_prediction/d021_g' + str(group) + '.csv')
    base = df_local['real'].max() # For normalization

    df_train['load' + str(group) + '_real'] = df_local['real'][period_status == 1] / base
    df_validation['load' + str(group) + '_real'] = df_local['real'][period_status == 2] / base
    df_test['load' + str(group) + '_real'] = df_local['real'][period_status == 3] / base

    df_train['load' + str(group) + '_local'] = df_local['prediction'][period_status == 1] / base
    df_validation['load' + str(group) + '_local'] = df_local['prediction'][period_status == 2] / base
    df_test['load' + str(group) + '_local'] = df_local['prediction'][period_status == 3] / base

    df_HFL = pd.read_csv('./data/processed/load_prediction/d022_g' + str(group) + '.csv')
    df_train['load' + str(group) + '_HFL'] = df_HFL['prediction'][period_status == 1] / base
    df_validation['load' + str(group) + '_HFL'] = df_HFL['prediction'][period_status == 2] / base
    df_test['load' + str(group) + '_HFL'] = df_HFL['prediction'][period_status == 3] / base

    df_VFL = pd.read_csv('./data/processed/load_prediction/d023_g' + str(group) + '.csv')
    df_train['load' + str(group) + '_VFL'] = df_VFL['prediction'][period_status == 1] / base
    df_validation['load' + str(group) + '_VFL'] = df_VFL['prediction'][period_status == 2] / base
    df_test['load' + str(group) + '_VFL'] = df_VFL['prediction'][period_status == 3] / base

## Renewables
df_renewable = pd.read_csv('./data/processed/renewable_data/d012.csv')

for i in range(num_wind):
    df_train['wind' + str(i) + '_real'] = df_renewable['wind' + str(i) + '_real'][period_status == 1]
    df_validation['wind' + str(i) + '_real'] = df_renewable['wind' + str(i) + '_real'][period_status == 2]
    df_test['wind' + str(i) + '_real'] = df_renewable['wind' + str(i) + '_real'][period_status == 3]

## Output prediction data
df_train.to_csv('./data/processed/combination/d031_train.csv', index=False)
df_validation.to_csv('./data/processed/combination/d031_validation.csv', index=False)
df_test.to_csv('./data/processed/combination/d031_test.csv', index=False)

## Divide train into n1 and n2
num_day = len(df_train) // horizon
order = list(range(num_day))
random.shuffle(order)
np.savetxt('./data/processed/combination/d031_order_n1_n2.txt', np.array(order), fmt='%d')
df_train_random = df_train.copy()
for i in range(num_day):
    df_train_random[(horizon * order[i]):(horizon * (order[i] + 1))] = df_train[(horizon * i):(horizon * (i + 1))]

df_train_n1 = df_train_random[:(n1 * horizon)]
df_train_n1.to_csv('./data/processed/combination/d031_train_n1.csv', index=False)
df_train_n2 = df_train_random[(n1 * horizon):]
df_train_n2.to_csv('./data/processed/combination/d031_train_n2.csv', index=False)
