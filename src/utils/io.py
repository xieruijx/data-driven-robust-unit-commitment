import numpy as np

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
    def input_strategy(num_bus, index_u_l_predict, type_u_l, name_method='', folder_strategies='./results/strategies/'):
        """
        Input UC strategies
        """
        x_og = np.loadtxt(folder_strategies + str(num_bus) + '/x_og_' + type_u_l + str(index_u_l_predict) + name_method + '.txt')
        x_pg = np.loadtxt(folder_strategies + str(num_bus) + '/x_pg_' + type_u_l + str(index_u_l_predict) + name_method + '.txt')
        x_rp = np.loadtxt(folder_strategies + str(num_bus) + '/x_rp_' + type_u_l + str(index_u_l_predict) + name_method + '.txt')
        x_rn = np.loadtxt(folder_strategies + str(num_bus) + '/x_rn_' + type_u_l + str(index_u_l_predict) + name_method + '.txt')
        y_rp = np.loadtxt(folder_strategies + str(num_bus) + '/y_rp_' + type_u_l + str(index_u_l_predict) + name_method + '.txt')
        y_rn = np.loadtxt(folder_strategies + str(num_bus) + '/y_rn_' + type_u_l + str(index_u_l_predict) + name_method + '.txt')
        return x_og, x_pg, x_rp, x_rn, y_rp, y_rn
    