import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

class IO(object):
    """
    IO class for input and output
    """

    @staticmethod
    def output_UC(num_bus, index_u_l_predict, type_u_l, weight, train_cost, train_order, validation_cost, test_cost, LBUB1, LBUB2, time, interpret, name_method='', folder_outputs='./results/outputs/', folder_strategies='./results/strategies/'):
        """
        Output UC results
        """
        np.savetxt(folder_strategies + str(num_bus) + '/weight_' + type_u_l + str(index_u_l_predict) + name_method + '.txt', weight)

        np.savetxt(folder_outputs + str(num_bus) + '/train_cost_' + type_u_l + str(index_u_l_predict) + name_method + '.txt', train_cost)
        np.savetxt(folder_outputs + str(num_bus) + '/train_order_' + type_u_l + str(index_u_l_predict) + name_method + '.txt', train_order)
        np.savetxt(folder_outputs + str(num_bus) + '/validation_cost_' + type_u_l + str(index_u_l_predict) + name_method + '.txt', validation_cost)
        np.savetxt(folder_outputs + str(num_bus) + '/test_cost_' + type_u_l + str(index_u_l_predict) + name_method + '.txt', test_cost)
        np.savetxt(folder_outputs + str(num_bus) + '/LBUB1_' + type_u_l + str(index_u_l_predict) + name_method + '.txt', LBUB1)
        np.savetxt(folder_outputs + str(num_bus) + '/LBUB2_' + type_u_l + str(index_u_l_predict) + name_method + '.txt', LBUB2)
        np.savetxt(folder_outputs + str(num_bus) + '/time_' + type_u_l + str(index_u_l_predict) + name_method + '.txt', time)

        np.savetxt(folder_strategies + str(num_bus) + '/x_og_' + type_u_l + str(index_u_l_predict) + name_method + '.txt', interpret['x_og'])
        np.savetxt(folder_strategies + str(num_bus) + '/x_pg_' + type_u_l + str(index_u_l_predict) + name_method + '.txt', interpret['x_pg'])
        np.savetxt(folder_strategies + str(num_bus) + '/x_rp_' + type_u_l + str(index_u_l_predict) + name_method + '.txt', interpret['x_rp'])
        np.savetxt(folder_strategies + str(num_bus) + '/x_rn_' + type_u_l + str(index_u_l_predict) + name_method + '.txt', interpret['x_rn'])
        np.savetxt(folder_strategies + str(num_bus) + '/y_rp_' + type_u_l + str(index_u_l_predict) + name_method + '.txt', interpret['y_rp'])
        np.savetxt(folder_strategies + str(num_bus) + '/y_rn_' + type_u_l + str(index_u_l_predict) + name_method + '.txt', interpret['y_rn'])

    @staticmethod
    def read_strategy(num_bus, index_u_l_predict, type_u_l, name_method='', folder_strategies='./results/strategies/'):
        """
        Read UC strategies
        """
        x_og = np.loadtxt(folder_strategies + str(num_bus) + '/x_og_' + type_u_l + str(index_u_l_predict) + name_method + '.txt')
        x_pg = np.loadtxt(folder_strategies + str(num_bus) + '/x_pg_' + type_u_l + str(index_u_l_predict) + name_method + '.txt')
        x_rp = np.loadtxt(folder_strategies + str(num_bus) + '/x_rp_' + type_u_l + str(index_u_l_predict) + name_method + '.txt')
        x_rn = np.loadtxt(folder_strategies + str(num_bus) + '/x_rn_' + type_u_l + str(index_u_l_predict) + name_method + '.txt')
        y_rp = np.loadtxt(folder_strategies + str(num_bus) + '/y_rp_' + type_u_l + str(index_u_l_predict) + name_method + '.txt')
        y_rn = np.loadtxt(folder_strategies + str(num_bus) + '/y_rn_' + type_u_l + str(index_u_l_predict) + name_method + '.txt')
        return x_og, x_pg, x_rp, x_rn, y_rp, y_rn
    
    @staticmethod
    def organize_method(num_bus, index_u_l_predict, type_u_l, epsilon=0.05, name_method='', folder_outputs='./results/outputs/'):
        """
        Read the output of one method
        (train quantile, validation quantile, test quantile, objective, time)
        """
        train_cost = np.loadtxt(folder_outputs + str(num_bus) + '/train_cost_' + type_u_l + str(index_u_l_predict) + name_method + '.txt')
        train_q = train_cost[np.argsort(train_cost)[np.ceil((1 - epsilon) * train_cost.shape[0]).astype(int) - 1]]

        validation_cost = np.loadtxt(folder_outputs + str(num_bus) + '/validation_cost_' + type_u_l + str(index_u_l_predict) + name_method + '.txt')
        validation_q = validation_cost[np.argsort(validation_cost)[np.ceil((1 - epsilon) * validation_cost.shape[0]).astype(int) - 1]]

        test_cost = np.loadtxt(folder_outputs + str(num_bus) + '/test_cost_' + type_u_l + str(index_u_l_predict) + name_method + '.txt')
        test_q = test_cost[np.argsort(test_cost)[np.ceil((1 - epsilon) * test_cost.shape[0]).astype(int) - 1]]

        LBUB2 = np.loadtxt(folder_outputs + str(num_bus) + '/LBUB2_' + type_u_l + str(index_u_l_predict) + name_method + '.txt', ndmin=2)
        obj = LBUB2[-1, 0]

        time = np.loadtxt(folder_outputs + str(num_bus) + '/time_' + type_u_l + str(index_u_l_predict) + name_method + '.txt')
        time = np.sum(time)

        return [train_q, validation_q, test_q, obj, time]
    
    @staticmethod
    def organize_methods(num_bus, index_u_l_predict, type_u_l, epsilon=0.05, folder_outputs='./results/outputs/', methods=['P1', 'P2', 'Proposed', 'RO_max', 'RO_quantile', 'RO_data', 'SP_MILP', 'SP_approx']):
        """
        Read and organize the outputs of methods
        """
        outputs = {}
        for method in methods:
            output = IO().organize_method(num_bus, index_u_l_predict, type_u_l, epsilon, method, folder_outputs)
            outputs[method] = output

        df = pd.DataFrame(outputs, index=['train quantile', 'validation quantile', 'test quantile', 'objective', 'time']).T
        print(df)
        df.to_csv(folder_outputs + str(num_bus) + '/outputs.csv')
    
    @staticmethod
    def ellipse(A, B, C, R, x0, y0, num_points=400):
        """
        Calculate the lines for plotting an ellipse
        A (x - x0)^2 + B (x - x0) (y - y0) + C (y - y0)^2 <= R
        """
        # The range of x
        xm = np.sqrt(R / (A - B * B / 4 / C))
        # Values for x at which to compute y values for the plots:
        x = np.linspace(x0 - xm, x0 + xm, num_points)
        # yp, yn: The upper and lower parts
        yp = np.zeros(x.shape)
        yp[1:-1] = y0 + (- B * (x[1:-1] - x0) + np.sqrt(B * B * (x[1:-1] - x0) * (x[1:-1] - x0) - 4 * C * (A * (x[1:-1] - x0) * (x[1:-1] - x0) - R))) / 2 / C
        yp[0] = y0 + B * xm / 2 / C
        yp[-1] = y0 - B * xm / 2 / C
        yn = np.zeros(x.shape)
        yn[1:-1] = y0 + (- B * (x[1:-1] - x0) - np.sqrt(B * B * (x[1:-1] - x0) * (x[1:-1] - x0) - 4 * C * (A * (x[1:-1] - x0) * (x[1:-1] - x0) - R))) / 2 / C
        yn[0] = y0 + B * xm / 2 / C
        yn[-1] = y0 - B * xm / 2 / C
        return x, yp, yn
    
    @staticmethod
    def projection_ellipse(index_uncertainty, error_mu, error_sigma, error_rho, u_l_predict, b_sum=True, num_points=400):
        """
        Project the multi-dimensional ellipsoid to a two-dimensional ellipse
        """
        if b_sum:
            u_l = u_l_predict.reshape((24, -1)) - error_mu.reshape((24, -1))
            x0 = np.sum(u_l[:, index_uncertainty[0]])
            y0 = np.sum(u_l[:, index_uncertainty[1]])

            P = np.eye(error_mu.shape[0])
            for i in range(u_l.shape[1]):
                P[i, i + 1: 24: -1] = - 1
            sigma = P.T @ np.linalg.inv(error_sigma) @ P
        else:
            u_l = u_l_predict - error_mu
            x0 = u_l[index_uncertainty[0]]
            y0 = u_l[index_uncertainty[1]]

            sigma = np.linalg.inv(error_sigma)

        sigma11 = sigma[index_uncertainty, :][:, index_uncertainty]
        sigma12 = np.delete(sigma, index_uncertainty, axis=1)[index_uncertainty, :]
        sigma22 = np.delete(np.delete(sigma, index_uncertainty, axis=0), index_uncertainty, axis=1)

        sigma_p = sigma11 - sigma12 @ np.linalg.inv(sigma22) @ (sigma12.T)

        A = sigma_p[0, 0]
        B = sigma_p[0, 1] + sigma_p[1, 0]
        C = sigma_p[1, 1]

        x, yp, yn = IO().ellipse(A, B, C, error_rho, x0, y0, num_points)
        return x, yp, yn
    
    @staticmethod
    def projection_bound(index_uncertainty, error_lb, error_ub, u_lu, u_ll, u_l_predict, b_sum=True):
        """
        Project the multi-dimensional bounds to two dimensions
        """
        ul = np.maximum(u_ll, u_l_predict - error_ub)
        uu = np.minimum(u_lu, u_l_predict - error_lb)

        if b_sum:
            ul = ul.reshape((24, -1))
            uu = uu.reshape((24, -1))

            lb = np.sum(ul[:, index_uncertainty], axis=0)
            ub = np.sum(uu[:, index_uncertainty], axis=0)
        else:
            lb = ul[index_uncertainty]
            ub = uu[index_uncertainty]

        xlx = [lb[0], lb[0]]
        xly = [lb[1], ub[1]]
        xux = [ub[0], ub[0]]
        xuy = [lb[1], ub[1]]
        ylx = [lb[0], ub[0]]
        yly = [lb[1], lb[1]]
        yux = [lb[0], ub[0]]
        yuy = [ub[1], ub[1]]

        return xlx, xly, xux, xuy, ylx, yly, yux, yuy, lb, ub
    
    @staticmethod
    def ineq2vertex(pA, pB):
        """
        Input the 2-dimentional compact polyhedron expressed by pA p >= pB
        Output the vertices in shape (-1, 2)
        No abundant constraint
        """
        pAnorm = np.sqrt(pA[:, 0] * pA[:, 0] + pA[:, 1] * pA[:, 1])
        pA[:, 0] = pA[:, 0] / pAnorm
        pA[:, 1] = pA[:, 1] / pAnorm
        pB = pB / pAnorm

        pA1p = pA[pA[:, 1] >= 0, :]
        pB1p = pB[pA[:, 1] >= 0]
        pA1n = pA[pA[:, 1] < 0, :]
        pB1n = pB[pA[:, 1] < 0]
        index = np.argsort(pA1p[:, 0])[::-1]
        pA1p = pA1p[index, :]
        pB1p = pB1p[index]
        index = np.argsort(pA1n[:, 0])
        pA1n = pA1n[index, :]
        pB1n = pB1n[index]

        pAsort = np.concatenate((pA1p, pA1n), axis=0)
        pBsort = np.concatenate((pB1p, pB1n))

        num_con = pAsort.shape[0]
        
        vertices = np.zeros((num_con, 2))
        for i in range(num_con - 1):
            vertices[i, 0] = (pBsort[i] * pAsort[i + 1, 1] - pBsort[i + 1] * pAsort[i, 1]) / (pAsort[i, 0] * pAsort[i + 1, 1] - pAsort[i, 1] * pAsort[i + 1, 0])
            vertices[i, 1] = (pBsort[i] * pAsort[i + 1, 0] - pBsort[i + 1] * pAsort[i, 0]) / (pAsort[i, 1] * pAsort[i + 1, 0] - pAsort[i, 0] * pAsort[i + 1, 1])
        vertices[-1, 0] = (pBsort[-1] * pAsort[0, 1] - pBsort[1] * pAsort[-1, 1]) / (pAsort[-1, 0] * pAsort[0, 1] - pAsort[-1, 1] * pAsort[0, 0])
        vertices[-1, 1] = (pBsort[-1] * pAsort[0, 0] - pBsort[1] * pAsort[-1, 0]) / (pAsort[-1, 1] * pAsort[0, 0] - pAsort[-1, 0] * pAsort[0, 1])

        print(vertices)
        
        return vertices

    @staticmethod
    def projection_polyhedron(index_uncertainty, coefficients, pmin, pmax, b_sum=True):
        """
        Project the multi-dimensional polyhedron to a two-dimensional polyhedron
        """
        Aueu = coefficients['Aueu']
        Auey = coefficients['Auey']
        Auiy = coefficients['Auiy']
        Bue = coefficients['Bue']
        Bui = coefficients['Bui']

        dim_u = Aueu.shape[1]
        dim_y = Auey.shape[1]

        Eu = np.zeros((2, dim_u))
        if b_sum:
            Eu[0, index_uncertainty[0]:-1:24] = 1
            Eu[1, index_uncertainty[1]:-1:24] = 1
        else:
            Eu[0, index_uncertainty[0]] = 1
            Eu[1, index_uncertainty[1]] = 1

        pA = np.array([[1, 0],
                       [0, 1],
                       [-1, 0],
                       [0, -1]]) # pA p >= pB
        pB = np.array([pmin[0], pmin[1], -pmax[0], -pmax[1]])
        pB = np.array([-4, -3, -2, -1])
        
        while True:
            vertices = IO().ineq2vertex(pA, pB)

            num_v = len(vertices)
            distance = 0
            xv = vertices[0, :]
            v = vertices[0, :]
            for i in range(num_v):
                m = gp.Model('m')
                u = m.addMVar((dim_u,), lb=0, ub=1, vtype=GRB.CONTINUOUS)
                y = m.addMVar((dim_y,), lb=-float('inf'), vtype=GRB.CONTINUOUS)
                m.addConstr(Aueu @ u + Auey @ y == Bue, name='e')
                m.addConstr(Auiy @ y >= Bui, name='i')
                m.setObjective(gp.quicksum((Eu @ u - vertices[i, :]) * (Eu @ u - vertices[i, :])), GRB.MINIMIZE)
                m.optimize()
                if m.ObjVal > distance:
                    xv = Eu @ u.X
                    v = vertices[i, :]
                    distance = m.ObjVal
            if distance < 1e-5:
                break
            a = xv - v
            pA = np.concatenate((pA, - a.reshape((1, 2))), axis=0)
            pB = np.append(pB, - a @ xv)
            print(distance)
            print(vertices)
            print(pA)
            print(pB)

        return vertices