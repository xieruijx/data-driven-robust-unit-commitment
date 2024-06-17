import numpy as np
import pandas as pd

class IO(object):
    """
    IO class for input and output
    """

    @staticmethod
    def output_UC(index_u_l_predict, type_u_l, weight, train_cost, train_order, validation_cost, test_cost, LBUB1, LBUB2, time, interpret, name_method='', folder_outputs='./results/outputs/', folder_strategies='./results/strategies/'):
        """
        Output UC results
        """
        np.savetxt(folder_strategies + 'weight_' + type_u_l + str(index_u_l_predict) + name_method + '.txt', weight)

        np.savetxt(folder_outputs + 'train_cost_' + type_u_l + str(index_u_l_predict) + name_method + '.txt', train_cost)
        np.savetxt(folder_outputs + 'train_order_' + type_u_l + str(index_u_l_predict) + name_method + '.txt', train_order)
        np.savetxt(folder_outputs + 'validation_cost_' + type_u_l + str(index_u_l_predict) + name_method + '.txt', validation_cost)
        np.savetxt(folder_outputs + 'test_cost_' + type_u_l + str(index_u_l_predict) + name_method + '.txt', test_cost)
        np.savetxt(folder_outputs + 'LBUB1_' + type_u_l + str(index_u_l_predict) + name_method + '.txt', LBUB1)
        np.savetxt(folder_outputs + 'LBUB2_' + type_u_l + str(index_u_l_predict) + name_method + '.txt', LBUB2)
        np.savetxt(folder_outputs + 'time_' + type_u_l + str(index_u_l_predict) + name_method + '.txt', time)

        np.savetxt(folder_strategies + 'x_og_' + type_u_l + str(index_u_l_predict) + name_method + '.txt', interpret['x_og'])
        np.savetxt(folder_strategies + 'x_pg_' + type_u_l + str(index_u_l_predict) + name_method + '.txt', interpret['x_pg'])
        np.savetxt(folder_strategies + 'x_rp_' + type_u_l + str(index_u_l_predict) + name_method + '.txt', interpret['x_rp'])
        np.savetxt(folder_strategies + 'x_rn_' + type_u_l + str(index_u_l_predict) + name_method + '.txt', interpret['x_rn'])
        np.savetxt(folder_strategies + 'y_rp_' + type_u_l + str(index_u_l_predict) + name_method + '.txt', interpret['y_rp'])
        np.savetxt(folder_strategies + 'y_rn_' + type_u_l + str(index_u_l_predict) + name_method + '.txt', interpret['y_rn'])

    @staticmethod
    def read_strategy(index_u_l_predict, type_u_l, name_method='', folder_strategies='./results/strategies/'):
        """
        Read UC strategies
        """
        x_og = np.loadtxt(folder_strategies + 'x_og_' + type_u_l + str(index_u_l_predict) + name_method + '.txt')
        x_pg = np.loadtxt(folder_strategies + 'x_pg_' + type_u_l + str(index_u_l_predict) + name_method + '.txt')
        x_rp = np.loadtxt(folder_strategies + 'x_rp_' + type_u_l + str(index_u_l_predict) + name_method + '.txt')
        x_rn = np.loadtxt(folder_strategies + 'x_rn_' + type_u_l + str(index_u_l_predict) + name_method + '.txt')
        y_rp = np.loadtxt(folder_strategies + 'y_rp_' + type_u_l + str(index_u_l_predict) + name_method + '.txt')
        y_rn = np.loadtxt(folder_strategies + 'y_rn_' + type_u_l + str(index_u_l_predict) + name_method + '.txt')
        return x_og, x_pg, x_rp, x_rn, y_rp, y_rn
    
    @staticmethod
    def read_weight_cost(index_u_l_predict, type_u_l, epsilon=0.05, name_method='', folder_number='./data/processed/weight/', folder_outputs='./data/processed/weight/outputs/', folder_strategies='./data/processed/weight/strategies/'):
        """
        Read the output files of different weights
        """
        file_number = file_number = folder_number + 'number_' + type_u_l + str(index_u_l_predict) + '.txt'
        numbers = pd.read_csv(file_number, header=None)
        numbers = numbers[0]
        number = numbers.iloc[-1]

        set_weight = np.zeros((number, 3))
        set_cost = np.zeros((number,))
        set_rate = np.zeros((number,))

        for i in range(number):
            weight = np.loadtxt(folder_strategies + 'n' + str(i) + '_weight_' + type_u_l + str(index_u_l_predict) + name_method + '.txt')
            set_weight[i, :] = weight
            cost = np.loadtxt(folder_outputs + 'n' + str(i) + '_validation_cost_' + type_u_l + str(index_u_l_predict) + name_method + '.txt')
            set_cost[i] = cost[np.argsort(cost)[np.ceil((1 - epsilon) * cost.shape[0]).astype(int) - 1]]
            set_rate[i] = - np.sum(np.isinf(cost)) / cost.shape[0] + 1

        outputs = {
            'w0': set_weight[:, 0],
            'w1': set_weight[:, 1],
            'w2': set_weight[:, 2],
            'cost': set_cost,
            'rate': set_rate
        }
        df = pd.DataFrame(outputs)
        print(df)
        df.to_csv(folder_outputs + 'outputs_' + type_u_l + str(index_u_l_predict) + '.csv')

        best_index = np.argsort(set_cost)[0]
        best_weight = set_weight[best_index, :]
        best_cost = set_cost[best_index]
        return df, best_index, best_weight, best_cost
    
    @staticmethod
    def read_training_data(type_u_l, u_select, epsilon=0.05, name_method='', file_numbers='./data/processed/weight/d056_number.txt', folder_outputs='./data/processed/weight/outputs/', folder_strategies='./data/processed/weight/strategies/', num_groups=21, num_wind=4):
        """
        Read training samples
        """
        with open(file_numbers, 'r') as f:
            numbers = [int(line.strip()) for line in f]
        sum_numbers = sum(numbers)

        matrix_load = np.zeros((24 * sum(u_select[:num_groups] * 3), sum_numbers))
        matrix_wind = np.zeros((24 * sum(u_select[num_groups:]), sum_numbers))
        matrix_weight = np.zeros((2, sum_numbers))
        matrix_cost = np.zeros((sum_numbers,))

        index_matrix = 0
        for index_u_l_predict in range(len(numbers)):
            for index_sample in range(numbers[index_u_l_predict]):
                predict_load, predict_wind, weight, cost = IO().read_training_sample(index_u_l_predict, type_u_l, index_sample, u_select, epsilon=epsilon, name_method=name_method, folder_outputs=folder_outputs, folder_strategies=folder_strategies, num_groups=num_groups, num_wind=num_wind)

                matrix_load[:, index_matrix] = predict_load
                matrix_wind[:, index_matrix] = predict_wind
                matrix_weight[:, index_matrix] = weight
                matrix_cost[index_matrix] = cost

                index_matrix = index_matrix + 1
        
        np.savetxt(folder_outputs + 'matrix_load.txt', matrix_load)
        np.savetxt(folder_outputs + 'matrix_wind.txt', matrix_wind)
        np.savetxt(folder_outputs + 'matrix_weight.txt', matrix_weight)
        np.savetxt(folder_outputs + 'matrix_cost.txt', matrix_cost)
        return matrix_load, matrix_wind, matrix_weight, matrix_cost

    @staticmethod
    def read_training_sample(index_u_l_predict, type_u_l, index_sample, u_select, epsilon=0.05, name_method='', folder_outputs='./data/processed/weight/outputs/', folder_strategies='./data/processed/weight/strategies/', num_groups=21, num_wind=4):
        """
        Read one training sample
        """

        data_load = np.zeros((24, num_groups, 3))
        data_wind = np.zeros((24, num_wind))
        df = pd.read_csv('./data/processed/combination/d031_' + type_u_l + '.csv')
        for group in range(num_groups):
            data_load[:, group, 0] = df['load' + str(group) + '_local'][(index_u_l_predict * 24):((index_u_l_predict + 1) * 24)].to_numpy()
            data_load[:, group, 1] = df['load' + str(group) + '_HFL'][(index_u_l_predict * 24):((index_u_l_predict + 1) * 24)].to_numpy()
            data_load[:, group, 2] = df['load' + str(group) + '_VFL'][(index_u_l_predict * 24):((index_u_l_predict + 1) * 24)].to_numpy()
        for wind in range(num_wind):
            data_wind[:, wind] = df['wind' + str(wind) + '_predict'][(index_u_l_predict * 24):((index_u_l_predict + 1) * 24)].to_numpy()
        predict_load = data_load[:, u_select[:num_groups]].reshape((-1,))
        predict_wind = data_wind[:, u_select[num_groups:]].reshape((-1,))

        weight = np.loadtxt(folder_strategies + 'n' + str(index_sample) + '_weight_' + type_u_l + str(index_u_l_predict) + name_method + '.txt')[:2]

        cost = np.loadtxt(folder_outputs + 'n' + str(index_sample) + '_validation_cost_' + type_u_l + str(index_u_l_predict) + name_method + '.txt')
        cost = cost[np.argsort(cost)[np.ceil((1 - epsilon) * cost.shape[0]).astype(int) - 1]]

        return predict_load, predict_wind, weight, cost
    
    @staticmethod
    def organize_method(index_u_l_predict, type_u_l, epsilon=0.05, name_method='', folder_outputs='./results/outputs/'):
        """
        Read the output of one method
        (train quantile, validation quantile, test quantile, objective, time)
        """
        train_cost = np.loadtxt(folder_outputs + 'train_cost_' + type_u_l + str(index_u_l_predict) + name_method + '.txt')
        train_q = train_cost[np.argsort(train_cost)[np.ceil((1 - epsilon) * train_cost.shape[0]).astype(int) - 1]]
        train_r = - np.sum(np.isinf(train_cost)) / train_cost.shape[0] + 1

        validation_cost = np.loadtxt(folder_outputs + 'validation_cost_' + type_u_l + str(index_u_l_predict) + name_method + '.txt')
        validation_q = validation_cost[np.argsort(validation_cost)[np.ceil((1 - epsilon) * validation_cost.shape[0]).astype(int) - 1]]
        validation_r = - np.sum(np.isinf(validation_cost)) / validation_cost.shape[0] + 1

        test_cost = np.loadtxt(folder_outputs + 'test_cost_' + type_u_l + str(index_u_l_predict) + name_method + '.txt')
        test_q = test_cost[np.argsort(test_cost)[np.ceil((1 - epsilon) * test_cost.shape[0]).astype(int) - 1]]
        test_r = - np.sum(np.isinf(test_cost)) / test_cost.shape[0] + 1

        if type_u_l == 'train':
            cost = train_cost[index_u_l_predict]
        elif type_u_l == 'validation':
            cost = validation_cost[index_u_l_predict]
        elif type_u_l == 'test':
            cost = test_cost[index_u_l_predict]
        else:
            raise RuntimeError('The type of u_l (type_u_l) is wrong')

        LBUB2 = np.loadtxt(folder_outputs + 'LBUB2_' + type_u_l + str(index_u_l_predict) + name_method + '.txt', ndmin=2)
        obj = LBUB2[-1, 0]

        time = np.loadtxt(folder_outputs + 'time_' + type_u_l + str(index_u_l_predict) + name_method + '.txt')
        time = np.sum(time)

        return [train_q, train_r, validation_q, validation_r, test_q, test_r, cost, obj, time]
    
    @staticmethod
    def organize_methods(index_u_l_predict, type_u_l, epsilon=0.05, folder_outputs='./results/outputs/', methods=['P1', 'P2', 'Proposed', 'RO_max', 'RO_quantile', 'SP_approx']):
        """
        Read and organize the outputs of methods
        """
        outputs = {}
        for method in methods:
            output = IO().organize_method(index_u_l_predict, type_u_l, epsilon, method, folder_outputs)
            outputs[method] = output

        df = pd.DataFrame(outputs, index=['train quantile', 'train rate', 'validation quantile', 'validation rate', 'test quantile', 'test rate', 'cost', 'objective', 'time']).T
        print(df[['test quantile', 'test rate', 'cost', 'objective', 'time']])
        df.to_csv(folder_outputs + 'outputs_' + type_u_l + str(index_u_l_predict) + '.csv')
        return df
    