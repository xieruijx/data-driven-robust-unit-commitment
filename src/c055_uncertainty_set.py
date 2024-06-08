## Draw the projections of uncertainty sets under different methods
import matplotlib.pyplot as plt
import numpy as np

from utils.optimization import Optimization
from utils.case import Case
from utils.io import IO

optimization = Optimization()

## Settings
index_u_l_predict = 0
type_u_l = 'test'

parameter = Case().case_ieee30_parameter()
# parameter['u_select'] = [False, True, True, False, False, False, False,
#                     False, False, False, False, False, False, False,
#                     False, False, False, False, False, False, False,
#                     False, False, False, False] # Only 2 loads are uncertain
parameter_epsilon0 = Case().case_ieee30_parameter(epsilon=0)
# parameter_epsilon0['u_select'] = [False, True, True, False, False, False, False,
#                     False, False, False, False, False, False, False,
#                     False, False, False, False, False, False, False,
#                     False, False, False, False] # Only 2 loads are uncertain

index_projection = [1, 2] # load 0-13, wind 14-15; project to the sum of the load across time periods

# weight_optimize = np.loadtxt('./data/processed/weight/index_9_weight_56.txt')
weight_optimize = np.loadtxt('./data/processed/combination/d032_weight.txt')
weight_error = np.loadtxt('./data/processed/combination/d032_weight.txt')

## P1: The data-driven RO method using the uncertainty set without reconstruction
error_mu, error_sigma, error_rho, u_l_predict, error_lb, error_ub = optimization.weight2ellipsoid(parameter, weight_optimize, 'n2', index_u_l_predict, 'case_ieee30', type_u_l)
x_P1, yp_P1, yn_P1 = IO().projection_sum(index_projection, error_mu, error_sigma, error_rho, u_l_predict, num_points=400)

## RO_max: The data-driven RO method using the 100% ellipsoidal uncertainty set
error_mu, error_sigma, error_rho, u_l_predict, error_lb, error_ub = optimization.weight2ellipsoid(parameter, weight_optimize, 'n_m', index_u_l_predict, 'case_ieee30', type_u_l)
x_RO_max, yp_RO_max, yn_RO_max = IO().projection_sum(index_projection, error_mu, error_sigma, error_rho, u_l_predict, num_points=400)

## RO_quantile: The data-driven RO method using the 1 - epsilon ellipsoidal uncertainty set
error_mu, error_sigma, error_rho, u_l_predict, error_lb, error_ub = optimization.weight2ellipsoid(parameter, weight_optimize, 'n_q', index_u_l_predict, 'case_ieee30', type_u_l)
x_RO_quantile, yp_RO_quantile, yn_RO_quantile = IO().projection_sum(index_projection, error_mu, error_sigma, error_rho, u_l_predict, num_points=400)

# Plotting:
fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=80)
ax.plot(x_P1, yp_P1, 'r', label="P1")
ax.plot(x_P1, yn_P1, 'r')
ax.plot(x_RO_max, yp_RO_max, 'g', label="RO_max")
ax.plot(x_RO_max, yn_RO_max, 'g')
ax.plot(x_RO_quantile, yp_RO_quantile, 'b', label="RO_quantile")
ax.plot(x_RO_quantile, yn_RO_quantile, 'b')
ax.legend()

# Set the extra space around the edges of the plot to zero:
ax.set_aspect('equal')

# Set the labels for the x and y axes:
ax.set_xlabel("X", fontsize=14)
ax.set_ylabel("Y", fontsize=14)

# Finally, show the plot:
plt.show()