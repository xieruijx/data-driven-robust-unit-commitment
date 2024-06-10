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

# b_sum = True # True for sum across periods, False for one period
# index_uncertainty = [0, 1] # b_sum = True; load 0-13, wind 14-15
b_sum = False
index_uncertainty = [160, 161]

# weight_optimize = np.loadtxt('./data/processed/weight/index_9_weight_56.txt')
weight_optimize = np.loadtxt('./data/processed/combination/d032_weight.txt')
weight_error = np.loadtxt('./data/processed/combination/d032_weight.txt')

## Upper and lower bounds under the common weight
error_mu, error_sigma, error_rho, u_l_predict, error_lb, error_ub, u_lu, u_ll = optimization.weight2ellipsoid(parameter, weight_optimize, 'n2', index_u_l_predict, 'case_ieee30', type_u_l)
xlx, xly, xux, xuy, ylx, yly, yux, yuy, pmin, pmax = IO().projection_bound(index_uncertainty, error_lb, error_ub, u_lu, u_ll, u_l_predict, b_sum)

## The first uncertainty set of Proposed
error_mu, error_sigma, error_rho, u_l_predict, error_lb, error_ub, u_lu, u_ll = optimization.weight2ellipsoid(parameter, weight_optimize, 'n1', index_u_l_predict, 'case_ieee30', type_u_l)
x_Proposed1, yp_Proposed1, yn_Proposed1 = IO().projection_ellipse(index_uncertainty, error_mu, error_sigma, error_rho, u_l_predict, b_sum, num_points=400)

## P1: The data-driven RO method using the uncertainty set without reconstruction
error_mu, error_sigma, error_rho, u_l_predict, error_lb, error_ub, u_lu, u_ll = optimization.weight2ellipsoid(parameter, weight_optimize, 'n2', index_u_l_predict, 'case_ieee30', type_u_l)
x_P1, yp_P1, yn_P1 = IO().projection_ellipse(index_uncertainty, error_mu, error_sigma, error_rho, u_l_predict, b_sum, num_points=400)

## RO_max: The data-driven RO method using the 100% ellipsoidal uncertainty set
error_mu, error_sigma, error_rho, u_l_predict, error_lb, error_ub, u_lu, u_ll = optimization.weight2ellipsoid(parameter, weight_optimize, 'n_m', index_u_l_predict, 'case_ieee30', type_u_l)
x_RO_max, yp_RO_max, yn_RO_max = IO().projection_ellipse(index_uncertainty, error_mu, error_sigma, error_rho, u_l_predict, b_sum, num_points=400)

## RO_quantile: The data-driven RO method using the 1 - epsilon ellipsoidal uncertainty set
error_mu, error_sigma, error_rho, u_l_predict, error_lb, error_ub, u_lu, u_ll = optimization.weight2ellipsoid(parameter, weight_optimize, 'n_q', index_u_l_predict, 'case_ieee30', type_u_l)
x_RO_quantile, yp_RO_quantile, yn_RO_quantile = IO().projection_ellipse(index_uncertainty, error_mu, error_sigma, error_rho, u_l_predict, b_sum, num_points=400)

## P2: The data-driven RO method using the weight optimized by minimizing the error measure
# coefficients = optimization.weight2polyhedron(parameter, weight_error, index_u_l_predict, 'case_ieee30', type_u_l)
# Aueu = coefficients['Aueu'].todense()
# Auey = coefficients['Auey'].todense()
# Auiy = coefficients['Auiy'].todense()
# Bue = coefficients['Bue']
# Bui = coefficients['Bui']
# np.save('./data/temp/Aueu_P2.npy', Aueu)
# np.save('./data/temp/Auey_P2.npy', Auey)
# np.save('./data/temp/Auiy_P2.npy', Auiy)
# np.save('./data/temp/Bue_P2.npy', Bue)
# np.save('./data/temp/Bui_P2.npy', Bui)
coefficients = {}
coefficients['Aueu'] = np.load('./data/temp/Aueu_P2.npy')
coefficients['Auey'] = np.load('./data/temp/Auey_P2.npy')
coefficients['Auiy'] = np.load('./data/temp/Auiy_P2.npy')
coefficients['Bue'] = np.load('./data/temp/Bue_P2.npy')
coefficients['Bui'] = np.load('./data/temp/Bui_P2.npy')
vertices = IO().projection_polyhedron(index_uncertainty, coefficients, pmin, pmax, b_sum)

# Plotting:
fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=80)
ax.plot(xlx, xly, 'k', label="Bound")
ax.plot(xux, xuy, 'k')
ax.plot(ylx, yly, 'k')
ax.plot(yux, yuy, 'k')
ax.plot(x_Proposed1, yp_Proposed1, 'y', label='Proposed1')
ax.plot(x_Proposed1, yn_Proposed1, 'y')
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
