## Draw the projections of uncertainty sets under different methods
import matplotlib.pyplot as plt
import numpy as np

from utils.optimization import Optimization
from utils.case import Case
from utils.projection import Project

optimization = Optimization()

## Settings
index_u_l_predict = 0
type_u_l = 'test'

parameter = Case().case_ieee30_parameter()
parameter['u_select'] = [False, True, True, False, False, False, False,
                    False, False, False, False, False, False, False,
                    False, False, False, False, False, False, False,
                    False, False, False, False] # Only 2 loads are uncertain
# parameter['TimeLimitFC'] = 100 # Time limit of the feasibility check problem
# parameter['TimeLimitSP'] = 100 # Time limit of the subproblem
parameter_epsilon0 = Case().case_ieee30_parameter(epsilon=0)
parameter_epsilon0['u_select'] = [False, True, True, False, False, False, False,
                    False, False, False, False, False, False, False,
                    False, False, False, False, False, False, False,
                    False, False, False, False] # Only 2 loads are uncertain
# parameter_epsilon0['TimeLimitFC'] = 100 # Time limit of the feasibility check problem
# parameter_epsilon0['TimeLimitSP'] = 100 # Time limit of the subproblem

Eu = np.zeros((2, sum(parameter['u_select']) * parameter['horizon']))
Eu[0, 0:14:2] = 1
Eu[1, 1:14:2] = 1

# weight_optimize = np.loadtxt('./data/processed/weight/index_9_weight_56.txt')
weight_optimize = np.loadtxt('./data/processed/combination/d032_weight.txt')
weight_error = np.loadtxt('./data/processed/combination/d032_weight.txt')

## Upper and lower bounds under the common weight
error_mu, error_sigma, error_rho, u_l_predict, error_lb, error_ub, u_lu, u_ll = optimization.weight2ellipsoid(parameter, weight_optimize, 'n2', index_u_l_predict, 'case_ieee30', type_u_l)
xlx, xly, xux, xuy, ylx, yly, yux, yuy, pmin, pmax = Project().projection_bound(error_lb, error_ub, u_lu, u_ll, u_l_predict, Eu)

## The first uncertainty set of Proposed
error_mu, error_sigma, error_rho, u_l_predict, error_lb, error_ub, u_lu, u_ll = optimization.weight2ellipsoid(parameter, weight_optimize, 'n1', index_u_l_predict, 'case_ieee30', type_u_l)
x_Proposed1, yp_Proposed1, yn_Proposed1 = Project().projection_ellipse(error_mu, error_sigma, error_rho, u_l_predict, Eu, num_points=400)

## P1: The data-driven RO method using the uncertainty set without reconstruction
error_mu, error_sigma, error_rho, u_l_predict, error_lb, error_ub, u_lu, u_ll = optimization.weight2ellipsoid(parameter, weight_optimize, 'n2', index_u_l_predict, 'case_ieee30', type_u_l)
x_P1, yp_P1, yn_P1 = Project().projection_ellipse(error_mu, error_sigma, error_rho, u_l_predict, Eu, num_points=400)

## RO_max: The data-driven RO method using the 100% ellipsoidal uncertainty set
error_mu, error_sigma, error_rho, u_l_predict, error_lb, error_ub, u_lu, u_ll = optimization.weight2ellipsoid(parameter, weight_optimize, 'n_m', index_u_l_predict, 'case_ieee30', type_u_l)
x_RO_max, yp_RO_max, yn_RO_max = Project().projection_ellipse(error_mu, error_sigma, error_rho, u_l_predict, Eu, num_points=400)

## RO_quantile: The data-driven RO method using the 1 - epsilon ellipsoidal uncertainty set
error_mu, error_sigma, error_rho, u_l_predict, error_lb, error_ub, u_lu, u_ll = optimization.weight2ellipsoid(parameter, weight_optimize, 'n_q', index_u_l_predict, 'case_ieee30', type_u_l)
x_RO_quantile, yp_RO_quantile, yn_RO_quantile = Project().projection_ellipse(error_mu, error_sigma, error_rho, u_l_predict, Eu, num_points=400)

## P2: The data-driven RO method using the weight optimized by minimizing the error measure
# coefficients, u_data_outside = optimization.weight2polyhedron(parameter, weight_error, index_u_l_predict, 'case_ieee30', type_u_l)
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
# np.save('./data/temp/u_data_outside_P2.npy', u_data_outside)

coefficients = {}
coefficients['Aueu'] = np.load('./data/temp/Aueu_P2.npy')
coefficients['Auey'] = np.load('./data/temp/Auey_P2.npy')
coefficients['Auiy'] = np.load('./data/temp/Auiy_P2.npy')
coefficients['Bue'] = np.load('./data/temp/Bue_P2.npy')
coefficients['Bui'] = np.load('./data/temp/Bui_P2.npy')
vertices = Project().projection_polyhedron(coefficients, pmin, pmax, Eu)
x_P2 = np.append(vertices[:, 0], vertices[0, 0])
y_P2 = np.append(vertices[:, 1], vertices[0, 1])

fontsize = 12
# Plotting:
fig, ax = plt.subplots(1, 1)
ax.plot(xlx, xly, 'k', label="Bound")
ax.plot(xux, xuy, 'k')
ax.plot(ylx, yly, 'k')
ax.plot(yux, yuy, 'k')
ax.plot(x_RO_max, yp_RO_max, 'g', label="RO1")
ax.plot(x_RO_max, yn_RO_max, 'g')
ax.plot(x_RO_quantile, yp_RO_quantile, 'b', label="RO2")
ax.plot(x_RO_quantile, yn_RO_quantile, 'b')
ax.plot(x_P1, yp_P1, 'r', label="P1")
ax.plot(x_P1, yn_P1, 'r')
ax.plot(x_Proposed1, yp_Proposed1, 'y', label='P2_1')
ax.plot(x_Proposed1, yn_Proposed1, 'y')
ax.plot(x_P2, y_P2, label='P2_2')
ax.legend()

# Set the extra space around the edges of the plot to zero:
ax.set_aspect('equal')

# Set the labels for the x and y axes:
ax.set_xlabel("First dimension (MW)", fontsize=fontsize)
ax.set_ylabel("Second dimension (MW)", fontsize=fontsize)

ax.grid(linestyle='--')

# Finally, show the plot:
plt.show()
