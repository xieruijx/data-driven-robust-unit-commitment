import numpy as np
import pandas as pd

from utils.combhandler import CombHandler

class C041(object):
    """
    C041 class for uncertainty initiation
    """

    @staticmethod
    def c041_initial_uncertainty(type_r, horizon, epsilon, delta, u_select, train_n1_real, train_n1_predict, train_n2_real, train_n2_predict):
        """
        Calculate the first uncertainty set
        type_r: 'n1' max in n1; 'n2' quantile in n2; 'n_m' max in n1 and n2; 'n_q' quantile in n1 and n2
        """
        print('(((((((((((((((((((((((((((((c041)))))))))))))))))))))))))))))')

        ## Select uncertain load and renewable
        train_n1_real = train_n1_real[:, u_select]
        train_n1_predict = train_n1_predict[:, u_select]
        train_n2_real = train_n2_real[:, u_select]
        train_n2_predict = train_n2_predict[:, u_select]

        error_n1 = train_n1_predict - train_n1_real
        error_n2 = train_n2_predict - train_n2_real

        ## Reshape into each sample
        n1 = train_n1_real.shape[0] // horizon
        n2 = train_n2_real.shape[0] // horizon

        error_n1 = error_n1.reshape((n1, -1))
        error_n2 = error_n2.reshape((n2, -1))

        dim_uncertainty = error_n1.shape[1]

        if type_r == 'n1' or type_r == 'n2':
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

            ## Find radius
            if type_r == 'n1':
                rho = np.diagonal(derror_n1 @ np.linalg.solve(sigma, derror_n1.T))
                radius = np.max(rho)
            else:
                rho = np.diagonal(derror_n2 @ np.linalg.solve(sigma, derror_n2.T))
                rank = CombHandler().get_rank(n2, epsilon, delta)
                radius = rho[np.argsort(rho)[rank - 1]]
        else:
            error = np.concatenate((error_n1, error_n2), axis=0)
            n = n1 + n2

            ## Calculate mean and covariance
            mu = error.mean(axis=0)
            derror = error - np.ones((n, 1)) @ mu.reshape((1, -1))
            sigma0 = derror.T @ derror / (n - 1)
            sigma = np.zeros((sigma0.shape))
            if np.linalg.matrix_rank(sigma0) < dim_uncertainty:
                for i in range(dim_uncertainty // horizon):
                    sigma[(i * horizon):((i + 1) * horizon), (i * horizon):((i + 1) * horizon)] = sigma0[(i * horizon):((i + 1) * horizon), (i * horizon):((i + 1) * horizon)]
            else:
                sigma = sigma0
            rho = np.diagonal(derror_n1 @ np.linalg.solve(sigma, derror_n1.T))

            ## Find radius
            if type_r == 'n_m':
                radius = np.max(rho)
            elif type_r == 'n_q':
                rank = np.ceil((1 - epsilon) * n).astype(int)
                radius = rho[np.argsort(rho)[rank - 1]]
            else:
                print('The type of radius is wrong.')
                return None, None, None
        print('Radius: {} (type: {})'.format(radius, type_r))

        return mu, sigma, radius
    