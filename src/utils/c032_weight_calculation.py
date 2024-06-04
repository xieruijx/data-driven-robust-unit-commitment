import numpy as np
import pandas as pd

class C032(object):
    """
    C032 class for weight calculation
    """
    
    @staticmethod
    def combine_predict(type_data, num_groups, weight):
        """
        Use the weight to generate the combined prediction
        """
        ## Input
        df = pd.read_csv('./data/processed/combination/d031_' + type_data + '.csv')

        ## Use the obtained weight to generate the combined prediction
        df_new = pd.DataFrame({})
        for group in range(num_groups):
            df_new['load' + str(group) + '_real'] = df['load' + str(group) + '_real']
            df_new['load' + str(group) + '_predict'] = np.concatenate((df['load' + str(group) + '_local'].to_numpy().reshape((-1, 1)), df['load' + str(group) + '_HFL'].to_numpy().reshape((-1, 1)), df['load' + str(group) + '_VFL'].to_numpy().reshape((-1, 1))), axis=1) @ weight
        df_new['wind1_real'] = df['wind1_real']
        df_new['wind1_predict'] = df['wind1_predict']
        df_new['wind2_real'] = df['wind2_real']
        df_new['wind2_predict'] = df['wind2_predict']

        return df_new
    
    @staticmethod
    def df2matrix(df, list_name):
        """
        Transform into matrix form
        """
        matrix_real = np.zeros((len(df), len(list_name)))
        matrix_predict = np.zeros((len(df), len(list_name)))
        for i in range(len(list_name)):
            matrix_real[:, i] = df[list_name[i] + '_real']
            matrix_predict[:, i] = df[list_name[i] + '_predict']
        return matrix_real, matrix_predict
    
    @staticmethod
    def error_bounds(list_real, list_predict):
        """
        Estimate the bounds of errors
        """
        real = list_real[0]
        predict = list_predict[0]
        error = predict - real
        lb = np.min(error, axis=0)
        ub = np.max(error, axis=0)
        for i in range(1, len(list_real)):
            real = list_real[i]
            predict = list_predict[i]
            error = predict - real
            lb = np.minimum(lb, np.min(error, axis=0))
            ub = np.maximum(ub, np.max(error, axis=0))
        error_bounds = np.concatenate((lb.reshape((-1, 1)), ub.reshape((-1, 1))), axis=1)
        return error_bounds
    
    @staticmethod
    def c032_calculate_weight(num_groups, weight):
        """
        Calculate the optimized weight by formula and then output combined predictions and errors
        """
        print('(((((((((((((((((((((((((((((c032)))))))))))))))))))))))))))))')

        c032 = C032()

        ## Combine prediction and calculate MSE
        df_train = c032.combine_predict('train', num_groups, weight)
        df_train_n1 = c032.combine_predict('train_n1', num_groups, weight)
        df_train_n2 = c032.combine_predict('train_n2', num_groups, weight)
        df_validation = c032.combine_predict('validation', num_groups, weight)
        df_test = c032.combine_predict('test', num_groups, weight)

        ## Transform dataframes into matrices. Columns are load0-load20, wind1, wind2. Rows are periods.
        list_name = ['load' + str(i) for i in range(num_groups)]
        list_name.extend(['wind1', 'wind2'])
        train_real, train_predict = c032.df2matrix(df_train, list_name)
        train_n1_real, train_n1_predict = c032.df2matrix(df_train_n1, list_name)
        train_n2_real, train_n2_predict = c032.df2matrix(df_train_n2, list_name)
        validation_real, validation_predict = c032.df2matrix(df_validation, list_name)
        test_real, test_predict = c032.df2matrix(df_test, list_name)

        ## Calculate error bounds
        list_real = [train_real, validation_real, test_real]
        list_predict = [train_predict, validation_predict, test_predict]
        error_bounds = c032.error_bounds(list_real, list_predict)

        return train_real, train_predict, train_n1_real, train_n1_predict, train_n2_real, train_n2_predict, validation_real, validation_predict, test_real, test_predict, error_bounds