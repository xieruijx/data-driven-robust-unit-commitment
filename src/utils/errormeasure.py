import numpy as np

class ErrorMeasure(object):
    """
    ErrorMeasure class for calculating different error measures
    """

    @staticmethod
    def mse(predict, real):
        """
        Calculate MSE
        Input: num_sample * horizon * num_groups
        Output: num_groups
        """
        error = real - predict
        mse = np.sum(np.sum(error * error, axis=0), axis=0) / error.shape[0] / error.shape[1]
        return mse
    
    @staticmethod
    def rmse(predict, real):
        """
        Calculate RMSE
        Input: num_sample * horizon * num_groups
        Output: num_groups
        """
        rmse = np.sqrt(ErrorMeasure().mse(predict, real))
        return rmse
    
    @staticmethod
    def mae(predict, real):
        """
        Calculate MAE
        Input: num_sample * horizon * num_groups
        Output: num_groups
        """
        error = real - predict
        mae = np.sum(np.sum(np.abs(error), axis=0), axis=0) / error.shape[0] / error.shape[1]
        return mae
    
    @staticmethod
    def mape(predict, real):
        """
        Calculate MAPE
        Input: num_sample * horizon * num_groups
        Output: num_groups
        """
        error = real - predict
        mape = np.sum(np.sum(np.abs(error) / real, axis=0), axis=0) / error.shape[0] / error.shape[1]
        return mape
    
    @staticmethod
    def mase(predict, real):
        """
        Calculate MASE
        Input: num_sample * horizon * num_groups
        Output: num_groups
        """
        error = real - predict
        error_naive = real[:, 1:, :] - real[:, :-1, :]
        fraction = np.mean(np.abs(error), axis=1) / np.mean(np.abs(error_naive), axis=1)
        mase = np.mean(fraction, axis=0)
        return mase
    
    @staticmethod
    def errors(predict, real):
        """
        Calculate errors: MSE, RMSE, MAE, MAPE, MASE
        Input: num_sample * horizon * num_groups
        Output: 5 * num_groups
        """
        EM = ErrorMeasure()
        errors = np.zeros((5, predict.shape[2]))
        errors[0, :] = EM.mse(predict, real)
        errors[1, :] = EM.rmse(predict, real)
        errors[2, :] = EM.mae(predict, real)
        errors[3, :] = EM.mape(predict, real)
        errors[4, :] = EM.mase(predict, real)
        return errors
    
    @staticmethod
    def errors_mean(predict, real):
        """
        Calculate mean errors: MSE, RMSE, MAE, MAPE, MASE
        Input: num_sample * horizon * num_groups
        Output: 5
        """
        errors = ErrorMeasure().errors(predict, real)
        mean = np.mean(errors, axis=1)
        return mean
