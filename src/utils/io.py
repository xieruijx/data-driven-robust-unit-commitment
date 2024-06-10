import numpy as np
import pandas as pd

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
    