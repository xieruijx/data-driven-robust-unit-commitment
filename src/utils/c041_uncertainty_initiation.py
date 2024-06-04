import numpy as np
import pandas as pd

from utils.combhandler import CombHandler

class C041(object):
    """
    C041 class for uncertainty initiation
    """

    @staticmethod
    def c041_initial_uncertainty(b_use_n2, horizon, epsilon, delta, u_select, train_n1_real, train_n1_predict, train_n2_real, train_n2_predict):
        """
        Calculate the first uncertainty set
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
        rho_n1 = np.diagonal(derror_n1 @ np.linalg.solve(sigma, derror_n1.T))
        radius_n1 = np.max(rho_n1)
        print('Radius n1: {}'.format(radius_n1))
        rho_n2 = np.diagonal(derror_n2 @ np.linalg.solve(sigma, derror_n2.T))
        rank_n2 = CombHandler().get_rank(n2, epsilon, delta)
        radius_n2 = rho_n2[np.argsort(rho_n2)[rank_n2 - 1]]
        print('Radius n2: {}'.format(radius_n2))

        if b_use_n2:
            radius = radius_n2 * 1.001
        else:
            radius = radius_n1
        print('Radius: {}'.format(radius))

        return mu, sigma, radius
    
    @staticmethod
    def c041_initial_uncertainty_RO(b_use_n2, horizon, epsilon, delta, u_select, train_n1_real, train_n1_predict, train_n2_real, train_n2_predict):
        """
        Calculate the first uncertainty set
        """
        print('((((((((((((((((((((((((((((c041-RO))))))))))))))))))))))))))))')

        ## Select uncertain load and renewable
        train_n1_real = train_n1_real[:, u_select]
        train_n1_predict = train_n1_predict[:, u_select]
        train_n2_real = train_n2_real[:, u_select]
        train_n2_predict = train_n2_predict[:, u_select]
        train_real = np.concatenate((train_n1_real, train_n2_real), axis=0)
        train_predict = np.concatenate((train_n1_predict, train_n2_predict), axis=0)

        error = train_predict - train_real

        ## Reshape into each sample
        n = train_real.shape[0] // horizon

        error = error.reshape((n, -1))

        dim_uncertainty = error.shape[1]

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

        ## Find radius
        rho = np.diagonal(derror @ np.linalg.solve(sigma, derror.T))
        # rank = np.ceil((1 - epsilon) * n).astype(int)
        # radius = rho[np.argsort(rho)[rank - 1]]
        radius = np.max(rho)
        print('Radius: {}'.format(radius))

        return mu, sigma, radius