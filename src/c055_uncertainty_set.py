## Draw the projections of uncertainty sets under different methods
import matplotlib.pyplot as plt
import numpy as np

from utils.optimization import Optimization
from utils.case import Case
from utils.projection import Project

np.random.seed(4)

optimization = Optimization()

## Settings
index_u_l_predict = 16
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
Eu[0, 0:10:2] = 1
Eu[1, 1:10:2] = 1
worst_u_RO1 = np.load('./data/temp/worst_u_RO1.npy')
worst_u_RO2 = np.load('./data/temp/worst_u_RO2.npy')
worst_u_P1 = np.load('./data/temp/worst_u_P1.npy')
worst_u_Proposed1 = np.load('./data/temp/worst_u_Proposed1.npy')
worst_u_Proposed2 = np.load('./data/temp/worst_u_Proposed2.npy')
error_mu = np.load('./data/temp/error_mu.npy')
Eu_RO1 = (worst_u_RO1 - error_mu) / np.linalg.norm(worst_u_RO1 - error_mu)
Eu_RO2 = (worst_u_RO2 - error_mu) / np.linalg.norm(worst_u_RO2 - error_mu)
Eu_P1 = (worst_u_P1 - error_mu) / np.linalg.norm(worst_u_P1 - error_mu)
Eu_Proposed1 = (worst_u_Proposed1 - error_mu) / np.linalg.norm(worst_u_Proposed1 - error_mu)
Eu_Proposed2 = (worst_u_Proposed2 - error_mu) / np.linalg.norm(worst_u_Proposed2 - error_mu)
Euold = Eu
# Eu[0, :] = 2.0109 * Euold[0, :] - 0.6350 * Euold[1, :]
Eu[1, :] = 0.6350 * Euold[0, :] + 2.0109 * Euold[1, :]
Eu[0, :] = Eu_RO1
# Eu[0, :] = np.random.rand(sum(parameter['u_select']) * parameter['horizon'])
# Eu[1, :] = np.random.rand(sum(parameter['u_select']) * parameter['horizon'])
# Eu[0, :] = Eu[0, :] / np.linalg.norm(Eu[0, :])
# Eu[1, :] = Eu[1, :] / np.linalg.norm(Eu[1, :])

weight_optimize = np.loadtxt('./data/processed/combination/d053_weight.txt')
weight_error = np.loadtxt('./data/processed/combination/d032_weight.txt')

## Upper and lower bounds under the common weight
error_mu, error_sigma, error_rho, u_l_predict, error_lb, error_ub, u_lu, u_ll = optimization.weight2ellipsoid(parameter, weight_error, 'n2', index_u_l_predict, 'case_ieee30', type_u_l)
xlx, xly, xux, xuy, ylx, yly, yux, yuy, pmin, pmax = Project().projection_bound(error_lb, error_ub, u_lu, u_ll, u_l_predict, Eu)

# ## The first uncertainty set of Proposed
# error_mu, error_sigma, error_rho, u_l_predict, error_lb, error_ub, u_lu, u_ll = optimization.weight2ellipsoid(parameter, weight_optimize, 'n1', index_u_l_predict, 'case_ieee30', type_u_l)
# x_Proposed1, yp_Proposed1, yn_Proposed1 = Project().projection_ellipse(error_mu, error_sigma, error_rho, u_l_predict, Eu, num_points=400)

# # coefficients, u_data_outside = optimization.weight2polyhedron(parameter, weight_optimize, index_u_l_predict, 'case_ieee30', type_u_l)
# # Aueu = coefficients['Aueu'].todense()
# # Auey = coefficients['Auey'].todense()
# # Auiy = coefficients['Auiy'].todense()
# # Bue = coefficients['Bue']
# # Bui = coefficients['Bui']
# # np.save('./data/temp/Aueu_Proposed.npy', Aueu)
# # np.save('./data/temp/Auey_Proposed.npy', Auey)
# # np.save('./data/temp/Auiy_Proposed.npy', Auiy)
# # np.save('./data/temp/Bue_Proposed.npy', Bue)
# # np.save('./data/temp/Bui_Proposed.npy', Bui)
# # np.save('./data/temp/u_data_outside_Proposed.npy', u_data_outside)

# coefficients = {}
# coefficients['Aueu'] = np.load('./data/temp/Aueu_Proposed.npy')
# coefficients['Auey'] = np.load('./data/temp/Auey_Proposed.npy')
# coefficients['Auiy'] = np.load('./data/temp/Auiy_Proposed.npy')
# coefficients['Bue'] = np.load('./data/temp/Bue_Proposed.npy')
# coefficients['Bui'] = np.load('./data/temp/Bui_Proposed.npy')
# vertices = Project().projection_polyhedron(coefficients, pmin, pmax, Eu)
# x_Proposed2 = np.append(vertices[:, 0], vertices[0, 0])
# y_Proposed2 = np.append(vertices[:, 1], vertices[0, 1])

## P1: The data-driven RO method using the uncertainty set without reconstruction
error_mu, error_sigma, error_rho, u_l_predict, error_lb, error_ub, u_lu, u_ll = optimization.weight2ellipsoid(parameter, weight_error, 'n2', index_u_l_predict, 'case_ieee30', type_u_l)
np.save('./data/temp/error_mu.npy', np.array(error_mu))
# worst_u_P1, worst_cost_P1 = optimization.weight2worstcase(parameter, weight_error, 'n2', index_u_l_predict, 'case_ieee30', type_u_l)
# np.save('./data/temp/worst_u_P1.npy', np.array(worst_u_P1))
# np.save('./data/temp/worst_cost_P1.npy', np.array(worst_cost_P1))
worst_u_P1 = np.load('./data/temp/worst_u_P1.npy')
worst_cost_P1 = np.load('./data/temp/worst_cost_P1.npy')
print('Worst-case cost: {}'.format(worst_cost_P1))
x_P1, yp_P1, yn_P1 = Project().projection_ellipse(error_mu, error_sigma, error_rho, u_l_predict, Eu, num_points=400)

## RO_max: The data-driven RO method using the 100% ellipsoidal uncertainty set
error_mu, error_sigma, error_rho, u_l_predict, error_lb, error_ub, u_lu, u_ll = optimization.weight2ellipsoid(parameter, weight_error, 'n_m', index_u_l_predict, 'case_ieee30', type_u_l)
# worst_u_RO1, worst_cost_RO1 = optimization.weight2worstcase(parameter, weight_error, 'n_m', index_u_l_predict, 'case_ieee30', type_u_l)
# np.save('./data/temp/worst_u_RO1.npy', np.array(worst_u_RO1))
# np.save('./data/temp/worst_cost_RO1.npy', np.array(worst_cost_RO1))
worst_u_RO1 = np.load('./data/temp/worst_u_RO1.npy')
worst_cost_RO1 = np.load('./data/temp/worst_cost_RO1.npy')
print('Worst-case cost: {}'.format(worst_cost_RO1))
x_RO_max, yp_RO_max, yn_RO_max = Project().projection_ellipse(error_mu, error_sigma, error_rho, u_l_predict, Eu, num_points=400)

## RO_quantile: The data-driven RO method using the 1 - epsilon ellipsoidal uncertainty set
error_mu, error_sigma, error_rho, u_l_predict, error_lb, error_ub, u_lu, u_ll = optimization.weight2ellipsoid(parameter, weight_error, 'n_q', index_u_l_predict, 'case_ieee30', type_u_l)
# worst_u_RO2, worst_cost_RO2 = optimization.weight2worstcase(parameter, weight_error, 'n_q', index_u_l_predict, 'case_ieee30', type_u_l)
# np.save('./data/temp/worst_u_RO2.npy', np.array(worst_u_RO2))
# np.save('./data/temp/worst_cost_RO2.npy', np.array(worst_cost_RO2))
worst_u_RO2 = np.load('./data/temp/worst_u_RO2.npy')
worst_cost_RO2 = np.load('./data/temp/worst_cost_RO2.npy')
print('Worst-case cost: {}'.format(worst_cost_RO2))
x_RO_quantile, yp_RO_quantile, yn_RO_quantile = Project().projection_ellipse(error_mu, error_sigma, error_rho, u_l_predict, Eu, num_points=400)

## P2: The data-driven RO method using the weight optimized by minimizing the error measure
error_mu, error_sigma, error_rho, u_l_predict, error_lb, error_ub, u_lu, u_ll = optimization.weight2ellipsoid(parameter, weight_error, 'n1', index_u_l_predict, 'case_ieee30', type_u_l)
# worst_u_Proposed, worst_cost_Proposed = optimization.weight2worstcase(parameter, weight_error, 'n1', index_u_l_predict, 'case_ieee30', type_u_l)
# np.save('./data/temp/worst_u_Proposed.npy', np.array(worst_u_Proposed))
# np.save('./data/temp/worst_cost_Proposed.npy', np.array(worst_cost_Proposed))
worst_u_Proposed = np.load('./data/temp/worst_u_Proposed.npy')
worst_cost_Proposed = np.load('./data/temp/worst_cost_Proposed.npy')
# worst_u_Proposed1 = worst_u_Proposed[0]
# worst_u_Proposed2 = worst_u_Proposed[1]
# worst_cost_Proposed1 = worst_cost_Proposed[0]
# worst_cost_Proposed2 = worst_cost_Proposed[1]
# np.save('./data/temp/worst_u_Proposed1.npy', np.array(worst_u_Proposed1))
# np.save('./data/temp/worst_cost_Proposed1.npy', np.array(worst_cost_Proposed1))
worst_u_Proposed1 = np.load('./data/temp/worst_u_Proposed1.npy')
worst_cost_Proposed1 = np.load('./data/temp/worst_cost_Proposed1.npy')
print('Worst-case cost: {}'.format(worst_cost_Proposed1))
x_P21, yp_P21, yn_P21 = Project().projection_ellipse(error_mu, error_sigma, error_rho, u_l_predict, Eu, num_points=400)

# np.save('./data/temp/worst_u_Proposed2.npy', np.array(worst_u_Proposed2))
# np.save('./data/temp/worst_cost_Proposed2.npy', np.array(worst_cost_Proposed2))
worst_u_Proposed2 = np.load('./data/temp/worst_u_Proposed2.npy')
worst_cost_Proposed2 = np.load('./data/temp/worst_cost_Proposed2.npy')
print('Worst-case cost: {}'.format(worst_cost_Proposed2))

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
print(x_P2)
print(y_P2)

fontsize = 12
# Plotting:
fig, ax = plt.subplots(1, 1)
ax.plot(xlx * 100, xly * 100, 'k', label="Bound")
ax.plot(xux * 100, xuy * 100, 'k')
ax.plot(ylx * 100, yly * 100, 'k')
ax.plot(yux * 100, yuy * 100, 'k')
ax.plot(x_RO_max * 100, yp_RO_max * 100, 'g', label="RO1")
ax.plot(x_RO_max * 100, yn_RO_max * 100, 'g')
worst_project_RO1 = Eu @ worst_u_RO1
print(worst_project_RO1)
ax.scatter(worst_project_RO1[0] * 100, worst_project_RO1[1] * 100, c='g', marker='*', s=70)
ax.plot(x_RO_quantile * 100, yp_RO_quantile * 100, 'b', label="RO2")
ax.plot(x_RO_quantile * 100, yn_RO_quantile * 100, 'b')
worst_project_RO2 = Eu @ worst_u_RO2
print(worst_project_RO2)
ax.scatter(worst_project_RO2[0] * 100, worst_project_RO2[1] * 100, c='b', marker='*', s=70)
ax.plot(x_P1 * 100, yp_P1 * 100, 'r', label="P1")
ax.plot(x_P1 * 100, yn_P1 * 100, 'r')
worst_project_P1 = Eu @ worst_u_P1
print(worst_project_P1)
ax.scatter(worst_project_P1[0] * 100, worst_project_P1[1] * 100, c='r', marker='*', s=70)
ax.plot(x_P21 * 100, yp_P21 * 100, 'y', label='Proposed_1')
ax.plot(x_P21 * 100, yn_P21 * 100, 'y')
worst_project_Proposed1 = Eu @ worst_u_Proposed1
print(worst_project_Proposed1)
ax.scatter(worst_project_Proposed1[0] * 100, worst_project_Proposed1[1] * 100, c='y', marker='*', s=70)
ax.plot(x_P2 * 100, y_P2 * 100, 'c', label='Proposed_2')
worst_project_Proposed2 = Eu @ worst_u_Proposed2
print(worst_project_Proposed2)
ax.scatter(worst_project_Proposed2[0] * 100, worst_project_Proposed2[1] * 100, c='c', marker='*', s=70)
# ax.plot(x_Proposed1 * 100, yp_Proposed1 * 100, 'c', label='Proposed_1')
# ax.plot(x_Proposed1 * 100, yn_Proposed1 * 100, 'c')
# ax.plot(x_Proposed2 * 100, y_Proposed2 * 100, label='Proposed_2')
ax.legend()

# Set the extra space around the edges of the plot to zero:
ax.set_aspect('equal')

# Set the labels for the x and y axes:
ax.set_xlabel("The first dimension (MW)", fontsize=fontsize)
ax.set_ylabel("The second dimension (MW)", fontsize=fontsize)

ax.grid(linestyle='--')

# Finally, show the plot:
plt.show()
