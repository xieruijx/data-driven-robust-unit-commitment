## Calculate the forecast errors
# Imports
import pandas as pd
import numpy as np

from utils.errormeasure import ErrorMeasure

# Error measure
EM = ErrorMeasure()

# Parameters
num_groups = 21
horizon = 24
num_test = 47
num_sample = num_test
index_begin = 11736

# Method: local
predict_local = np.zeros((num_sample, horizon, num_groups))
real = np.zeros((num_sample, horizon, num_groups))
for group in range(num_groups):
    df = pd.read_csv('./data/processed/load_prediction/d021_g' + str(group) + '.csv')
    predict_local[:, :, group] = np.reshape(df['prediction'][index_begin:(index_begin + horizon * num_test)], (num_sample, horizon))
    real[:, :, group] = np.reshape(df['real'][index_begin:(index_begin + horizon * num_test)], (num_sample, horizon))
local_errors = EM.errors(predict_local, real)
local_errors_mean = EM.errors_mean(predict_local, real)

# Method: HFL
predict_HFL = np.zeros((num_sample, horizon, num_groups))
for group in range(num_groups):
    df = pd.read_csv('./data/processed/load_prediction/d022_g' + str(group) + '.csv')
    predict_HFL[:, :, group] = np.reshape(df['prediction'][index_begin:(index_begin + horizon * num_test)], (num_sample, horizon))
HFL_errors = EM.errors(predict_HFL, real)
HFL_errors_mean = EM.errors_mean(predict_HFL, real)

# Method: VFL
predict_VFL = np.zeros((num_sample, horizon, num_groups))
for group in range(num_groups):
    df = pd.read_csv('./data/processed/load_prediction/d023_g' + str(group) + '.csv')
    predict_VFL[:, :, group] = np.reshape(df['prediction'][index_begin:(index_begin + horizon * num_test)], (num_sample, horizon))
VFL_errors = EM.errors(predict_VFL, real)
VFL_errors_mean = EM.errors_mean(predict_VFL, real)
# predict_VFL1 = 0.4 * predict_local + 0.6 * predict_VFL
# VFL_errors = EM.errors(predict_VFL1, real)
# VFL_errors_mean = EM.errors_mean(predict_VFL1, real)

# Method: naive
predict_naive = np.zeros((num_sample, horizon, num_groups))
for group in range(num_groups):
    df = pd.read_csv('./data/processed/load_prediction/d021_g' + str(group) + '.csv')
    predict_naive[:, :, group] = np.reshape(df['real'][index_begin - horizon:(index_begin + horizon * num_test - horizon)], (num_sample, horizon))
naive_errors = EM.errors(predict_naive, real)
naive_errors_mean = EM.errors_mean(predict_naive, real)

# Method: C1 - Combination using optimized weight
weight_C1 = np.loadtxt('./data/processed/combination/d032_weight.txt')
predict_C1 = weight_C1[0] * predict_local + weight_C1[1] * predict_HFL + weight_C1[2] * predict_VFL
C1_errors = EM.errors(predict_C1, real)
C1_errors_mean = EM.errors_mean(predict_C1, real)

# Method: C2_30 - Combination using weight in IEEE 30-bus
weight_C2_30 = np.loadtxt('./data/processed/combination/d053_weight.txt')
predict_C2_30 = weight_C2_30[0] * predict_local + weight_C2_30[1] * predict_HFL + weight_C2_30[2] * predict_VFL
C2_30_errors = EM.errors(predict_C2_30, real)
C2_30_errors_mean = EM.errors_mean(predict_C2_30, real)

# Method: C2_118 - Combination using weight in IEEE 118-bus
weight_C2_118 = np.loadtxt('./data/processed/combination/d063_weight.txt')
predict_C2_118 = weight_C2_118[0] * predict_local + weight_C2_118[1] * predict_HFL + weight_C2_118[2] * predict_VFL
C2_118_errors = EM.errors(predict_C2_118, real)
C2_118_errors_mean = EM.errors_mean(predict_C2_118, real)

errors = {
    'Naive': naive_errors_mean,
    'BiLSTM': local_errors_mean,
    'HFL': HFL_errors_mean,
    'VFL': VFL_errors_mean,
    'C1': C1_errors_mean,
    'C2_30': C2_30_errors_mean,
    'C2_180': C2_118_errors_mean
}
df_errors = pd.DataFrame(errors, index=['MSE', 'RMSE', 'MAE', 'MAPE', 'MASE']).T
print(df_errors)
df_errors.to_csv('./results/outputs/o071_errors.csv')

num_groups = 5
real = real[:, :, :num_groups]
predict_VFL = np.zeros((num_sample, horizon, num_groups))
for group in range(num_groups):
    df = pd.read_csv('./data/processed/load_prediction/d024_g' + str(group) + '.csv')
    predict_VFL[:, :, group] = np.reshape(df['prediction'][index_begin:(index_begin + horizon * num_test)], (num_sample, horizon))
VFL_errors = EM.errors(predict_VFL, real)
VFL_errors_mean = EM.errors_mean(predict_VFL, real)
print(VFL_errors_mean)

num_groups = 5
real = real[:, :, :num_groups]
predict_VFL = np.zeros((num_sample, horizon, num_groups))
for group in range(num_groups):
    df = pd.read_csv('./data/processed/load_prediction/d023_g' + str(group) + '.csv')
    predict_VFL[:, :, group] = np.reshape(df['prediction'][index_begin:(index_begin + horizon * num_test)], (num_sample, horizon))
VFL_errors = EM.errors(predict_VFL, real)
VFL_errors_mean = EM.errors_mean(predict_VFL, real)
print(VFL_errors_mean)