## Calculate the first uncertainty set

import numpy as np

from utils.combhandler import CombHandler
from utils.case import Case

## Settings
b_use_n2 = False # False: the radius is the largest one in n1. True: the radius is the rank one in n2.
horizon = 24
epsilon = 0.05 # chance constraint parameter
delta = 0.05 # probability guarantee parameter
# u_select = [True, True, True, True, True, True, True,
#             False, False, False, False, False, False, False,
#             False, False, False, False, False, False, False,
#             True, True] # Only a part of loads and renewables are uncertain
u_select = [False, True, True, False, False, False, True,
            False, True, True, True, True, True, True,
            True, False, True, True, True, True, False,
            True, True] # Only a part of loads and renewables are uncertain
EPS = 1e-8

## Select uncertain load and renewable
real_n1 = np.load('./data/processed/combination/d032_real_train_n1.npy')[:, u_select]
real_n2 = np.load('./data/processed/combination/d032_real_train_n2.npy')[:, u_select]
predict_n1 = np.load('./data/processed/combination/d032_predict_train_n1.npy')[:, u_select]
predict_n2 = np.load('./data/processed/combination/d032_predict_train_n2.npy')[:, u_select]

error_n1 = predict_n1 - real_n1
error_n2 = predict_n2 - real_n2

## Reshape into each sample
n1 = real_n1.shape[0] // horizon
n2 = real_n2.shape[0] // horizon

error_n1 = error_n1.reshape((n1, -1))
error_n2 = error_n2.reshape((n2, -1))

dim_uncertainty = error_n1.shape[1]

## Calculate mean and covariance
mu = error_n1.mean(axis=0)
derror_n1 = error_n1 - np.ones((n1, 1)) @ mu.reshape((1, -1))
derror_n2 = error_n2 - np.ones((n2, 1)) @ mu.reshape((1, -1))
sigma0 = derror_n1.T @ derror_n1 / (n1 - 1)
sigma = np.zeros((sigma0.shape))
if np.linalg.matrix_rank(sigma0) < dim_uncertainty:
    for i in range(dim_uncertainty // horizon):
        sigma[(i * horizon):((i + 1) * horizon), (i * horizon):((i + 1) * horizon)] = sigma0[(i * horizon):((i + 1) * horizon), (i * horizon):((i + 1) * horizon)]
else:
    sigma = sigma0

np.save('./data/processed/uncertainty/d041_mu.npy', mu)
np.save('./data/processed/uncertainty/d041_sigma.npy', sigma)

## Find radius
rho_n1 = np.diagonal(derror_n1 @ np.linalg.solve(sigma, derror_n1.T))
radius_n1 = np.max(rho_n1)
print('Radius n1: {}'.format(radius_n1))
rho_n2 = np.diagonal(derror_n2 @ np.linalg.solve(sigma, derror_n2.T))
rank_n2 = CombHandler().get_rank(n2, epsilon, delta)
radius_n2 = rho_n2[np.argsort(rho_n2)[rank_n2 - 1]]
print('Radius n2: {}'.format(radius_n2))

if b_use_n2:
    radius = radius_n2
else:
    radius = radius_n1

np.save('./data/processed/uncertainty/d041_radius.npy', radius)

## Preparing uncertainty data
case = Case()
mpc = case.case_ieee30_modified()
mpc = case.process_case(mpc)

def calculate_u_data(real, predict, mpc, EPS, b_ellipsoid):
    """
    Calculate the uncertainty data from real and predict data
    Revise it into the uncertainty set according to requirements
    """
    real = real[:, mpc['u_select']]
    predict = predict[:, mpc['u_select']]

    num_data = real.shape[0] // mpc['n_t']
    u_data = np.zeros((num_data, mpc['n_t'] * mpc['n_u']))
    for i in range(num_data):
        u_data[i, :] = mpc['u_l_predict'] - predict[(i * mpc['n_t']):((i + 1) * mpc['n_t'])].reshape((-1,)) + real[(i * mpc['n_t']):((i + 1) * mpc['n_t'])].reshape((-1,))

    # b_ellipsoid = True: Revise the point into the ellipsoid uncertainty set if it is not
    # b_ellipsoid = False: Revise the point to satisfy the bounds if it is not
    for i in range(num_data):
        if not case.test_u(u_data[i, :], mpc, b_print=True, b_ellipsoid=b_ellipsoid):
            print('Data {} is not in the uncertainty set.'.format(i))
            u_data[i, :] = case.revise_u(u_data[i, :], mpc, EPS, b_print=True, b_ellipsoid=True)

    # Test whether the points are in the ellipsoid uncertainty set
    num_not_in_uncertainty_set = 0
    for i in range(num_data):
        if not case.test_u(u_data[i, :], mpc, b_print=True, b_ellipsoid=True):
            print('Data {} is not in the ellipsoid uncertainty set.'.format(i))
            num_not_in_uncertainty_set += 1
    print('{} out of {} data are not in the ellipsoid uncertainty set.'.format(num_not_in_uncertainty_set, num_data))
    
    return u_data

print('Construct uncertainty data from train (ellipsoid)')
train_real = np.load('./data/processed/combination/d032_real_train.npy')
train_predict = np.load('./data/processed/combination/d032_predict_train.npy')
u_data_train = calculate_u_data(train_real, train_predict, mpc, EPS, b_ellipsoid=True)
np.save('./data/processed/combination/d041_u_data_train.npy', u_data_train)

print('Construct uncertainty data from train n2 (bound)')
train_n2_real = np.load('./data/processed/combination/d032_real_train_n2.npy')
train_n2_predict = np.load('./data/processed/combination/d032_predict_train_n2.npy')
u_data_train_n2 = calculate_u_data(train_n2_real, train_n2_predict, mpc, EPS, b_ellipsoid=False)
np.save('./data/processed/combination/d041_u_data_train_n2.npy', u_data_train_n2)

print('Construct uncertainty data from validation (bound)')
validation_real = np.load('./data/processed/combination/d032_real_validation.npy')
validation_predict = np.load('./data/processed/combination/d032_predict_validation.npy')
u_data_validation = calculate_u_data(validation_real, validation_predict, mpc, EPS, b_ellipsoid=False)
np.save('./data/processed/combination/d041_u_data_validation.npy', u_data_validation)

print('Construct uncertainty data from test (bound)')
test_real = np.load('./data/processed/combination/d032_real_test.npy')
test_predict = np.load('./data/processed/combination/d032_predict_test.npy')
u_data_test = calculate_u_data(test_real, test_predict, mpc, EPS, b_ellipsoid=False)
np.save('./data/processed/combination/d041_u_data_test.npy', u_data_test)
