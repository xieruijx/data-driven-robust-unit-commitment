import numpy as np
import gurobipy as gp
from gurobipy import GRB

class Project(object):
    """
    Project class for ellipsoid and polyhedron projection to 2-dimensional plane
    """

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
    def projection_ellipse(error_mu, error_sigma, error_rho, u_l_predict, Eu, num_points=400):
        """
        Project the multi-dimensional ellipsoid to a two-dimensional ellipse
        """
        dim_u = Eu.shape[1]

        u_l = u_l_predict - error_mu
        x0 = Eu[0, :] @ u_l
        y0 = Eu[1, :] @ u_l
        
        basis = np.zeros((dim_u, dim_u))
        basis[:2, :] = Eu
        basis_element = [0, 0]
        basis_element[0] = np.nonzero(Eu[0, :])[0][0]
        basis_element[1] = np.nonzero(Eu[1, :] - Eu[0, :] * Eu[1, basis_element[0]] / Eu[0, basis_element[0]])[0][0]
        element = 2
        for i in range(dim_u):
            if i == basis_element[0] or i == basis_element[1]:
                continue
            else:
                basis[element, i] = 1
                element = element + 1

        P = np.linalg.inv(basis)
        sigma = P.T @ np.linalg.inv(error_sigma) @ P

        sigma11 = sigma[:2, :][:, :2]
        sigma12 = np.delete(sigma, [0, 1], axis=1)[:2, :]
        sigma22 = np.delete(np.delete(sigma, [0, 1], axis=0), [0, 1], axis=1)

        sigma_p = sigma11 - sigma12 @ np.linalg.inv(sigma22) @ (sigma12.T)

        A = sigma_p[0, 0]
        B = sigma_p[0, 1] + sigma_p[1, 0]
        C = sigma_p[1, 1]

        x, yp, yn = Project().ellipse(A, B, C, error_rho, x0, y0, num_points)
        return x, yp, yn
    
    @staticmethod
    def projection_bound(error_lb, error_ub, u_lu, u_ll, u_l_predict, Eu):
        """
        Project the multi-dimensional bounds to two dimensions
        """
        ul = np.maximum(u_ll, u_l_predict - error_ub)
        uu = np.minimum(u_lu, u_l_predict - error_lb)

        Eup = np.maximum(Eu, 0)
        Eun = np.minimum(Eu, 0)

        lb = Eup @ ul + Eun @ uu
        ub = Eup @ uu + Eun @ ul

        xlx = np.array([lb[0], lb[0]])
        xly = np.array([lb[1], ub[1]])
        xux = np.array([ub[0], ub[0]])
        xuy = np.array([lb[1], ub[1]])
        ylx = np.array([lb[0], ub[0]])
        yly = np.array([lb[1], lb[1]])
        yux = np.array([lb[0], ub[0]])
        yuy = np.array([ub[1], ub[1]])

        return xlx, xly, xux, xuy, ylx, yly, yux, yuy, lb, ub
    
    @staticmethod
    def ineq_norm(pA, pB):
        """
        Input the 2-dimentional compact polyhedron expressed by pA p >= pB
        Normalize pA
        """
        pAnorm = np.sqrt(pA[:, 0] * pA[:, 0] + pA[:, 1] * pA[:, 1])
        pA[:, 0] = pA[:, 0] / pAnorm
        pA[:, 1] = pA[:, 1] / pAnorm
        pB = pB / pAnorm
        return pA, pB
    
    @staticmethod
    def ineq_polyhedron(pA, pB):
        """
        Input the 2-dimentional compact polyhedron expressed by pA p >= pB
        Delete the abundant constraint
        Output pA p >= pB
        """
        pA, pB = Project().ineq_norm(pA, pB)

        num_con = pA.shape[0] + 1
        while pA.shape[0] < num_con:
            num_con = pA.shape[0]
            for i in range(pA.shape[0]):
                pAn = np.delete(pA, i, axis=0)
                pBn = np.delete(pB, i)
                m = gp.Model('m')
                x = m.addMVar((2,), lb=-float('inf'), vtype=GRB.CONTINUOUS)
                p = m.addMVar((2,), lb=-float('inf'), vtype=GRB.CONTINUOUS)
                m.addConstr(pA[i, :] @ x == pB[i])
                m.addConstr(pAn @ p >= pBn)
                m.setObjective(gp.quicksum((x - p) * (x - p)), GRB.MINIMIZE)
                m.setParam('OutputFlag', 0) 
                m.optimize()
                if m.ObjVal > 1e-5:
                    pA = pAn
                    pB = pBn
                    break
        return pA, pB
    
    @staticmethod
    def ineq2vertex(pA, pB):
        """
        Input the 2-dimentional compact polyhedron expressed by pA p >= pB
        Output the vertices in shape (-1, 2)
        No abundant constraint
        """
        pA, pB = Project().ineq_norm(pA, pB)

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
        vertices[-1, 0] = (pBsort[-1] * pAsort[0, 1] - pBsort[0] * pAsort[-1, 1]) / (pAsort[-1, 0] * pAsort[0, 1] - pAsort[-1, 1] * pAsort[0, 0])
        vertices[-1, 1] = (pBsort[-1] * pAsort[0, 0] - pBsort[0] * pAsort[-1, 0]) / (pAsort[-1, 1] * pAsort[0, 0] - pAsort[-1, 0] * pAsort[0, 1])
        
        return vertices

    @staticmethod
    def projection_polyhedron(coefficients, pmin, pmax, Eu):
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

        pA = np.array([[1, 0],
                       [0, 1],
                       [-1, 0],
                       [0, -1]]) # pA p >= pB
        pB = np.array([pmin[0], pmin[1], -pmax[0], -pmax[1]])
        
        while True:
            pA, pB = Project().ineq_polyhedron(pA, pB)
            vertices = Project().ineq2vertex(pA, pB)

            num_v = len(vertices)
            distance = 0
            xv = vertices[0, :]
            v = vertices[0, :]
            for i in range(num_v):
                m = gp.Model('m')
                u = m.addMVar((dim_u,), lb=-float('inf'), vtype=GRB.CONTINUOUS)
                y = m.addMVar((dim_y,), lb=-float('inf'), vtype=GRB.CONTINUOUS)
                m.addConstr(Aueu @ u + Auey @ y == Bue, name='e')
                m.addConstr(Auiy @ y >= Bui, name='i')
                m.setObjective(gp.quicksum((Eu @ u - vertices[i, :]) * (Eu @ u - vertices[i, :])), GRB.MINIMIZE)
                m.setParam('OutputFlag', 0) 
                m.optimize()
                if m.ObjVal > distance:
                    xv = Eu @ u.X
                    v = vertices[i, :]
                    distance = m.ObjVal
            if distance < 1e-6:
                break
            a = xv - v
            a = a / np.sqrt(np.sum(a * a))
            pA = np.concatenate((pA, a.reshape((1, 2))), axis=0)
            pB = np.append(pB, a @ xv)
            print(distance)

        return vertices