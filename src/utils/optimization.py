import numpy as np

from utils.c032_weight_calculation import C032
from utils.c041_uncertainty_initiation import C041
from utils.c042_dispatch_model import C042
from utils.c043_CCG_ellipsoid import C043
from utils.c044_reconstruction import C044
from utils.c045_CCG_polyhedron import C045
from utils.c046_evaluation import C046
from utils.c047_SP_enumeration import C047

class Optimization(object):
    """
    Optimization class for the optimization process
    """
    
    @staticmethod
    def weight2cost(parameter, weight, index_u_l_predict=0, name_case='case_ieee30'):
        """
        Combine c032, c041-c046
        """

        b_faster = parameter['b_faster']
        type_r = parameter['type_r']
        b_display_SP = parameter['b_display_SP']
        num_groups = parameter['num_groups']
        num_wind = parameter['num_wind']
        horizon = parameter['horizon']
        epsilon = parameter['epsilon']
        delta = parameter['delta']
        MaxIter = parameter['MaxIter']
        LargeNumber = parameter['LargeNumber']
        Tolerance = parameter['Tolerance']
        TimeLimitFC = parameter['TimeLimitFC']
        TimeLimitSP = parameter['TimeLimitSP']
        EPS = parameter['EPS']
        u_select = parameter['u_select']

        train_real, train_predict, train_n1_real, train_n1_predict, train_n2_real, train_n2_predict, validation_real, validation_predict, test_real, test_predict, error_bounds = C032().c032_calculate_weight(num_groups, num_wind, weight)
        error_mu, error_sigma, error_rho = C041().c041_initial_uncertainty(type_r, horizon, epsilon, delta, u_select, train_n1_real, train_n1_predict, train_n2_real, train_n2_predict)
        u_l_predict = validation_predict[(index_u_l_predict * horizon):((index_u_l_predict + 1) * horizon)]
        mpc, coefficients, u_data_train, u_data_train_n2, u_data_validation, u_data_test, u_data_train_original = C042().c042_dispatch_model(u_select, error_mu, error_sigma, error_rho, error_bounds, EPS, train_real, train_predict, train_n2_real, train_n2_predict, validation_real, validation_predict, test_real, test_predict, u_l_predict, name_case)

        # _, sxb1, sxc1, LBUB1, time1 = C043().c043_CCG_n1_faster(LargeNumber, Tolerance, TimeLimitFC, TimeLimitSP, MaxIter, EPS, mpc, coefficients, u_data_train, b_display_SP)
        # _, u_data, coefficients = C044().c044_reconstruction(epsilon, delta, coefficients, u_data_train_n2, sxb1, sxc1, LBUB1)
        # _, _, sxb2, sxc2, LBUB2, time2, interpret = C045().c045_CCG_n2_faster(LargeNumber, Tolerance, TimeLimitFC, TimeLimitSP, MaxIter, EPS, mpc, coefficients, u_data)
        # validation_cost = C046().c046_evaluate_faster(u_data_validation, coefficients, sxb2, sxc2, LBUB2)
        # test_cost = C046().c046_evaluate_faster(u_data_test, coefficients, sxb2, sxc2, LBUB2)
        if not b_faster:
            _, sxb1, sxc1, LBUB1, time1 = C043().c043_CCG_n1(LargeNumber, Tolerance, TimeLimitFC, TimeLimitSP, MaxIter, EPS, mpc, coefficients, u_data_train, b_display_SP)
            _, u_data, coefficients = C044().c044_reconstruction(epsilon, delta, coefficients, u_data_train_n2, sxb1, sxc1, LBUB1)
            _, _, sxb2, sxc2, LBUB2, time2, interpret = C045().c045_CCG_n2(LargeNumber, Tolerance, TimeLimitFC, TimeLimitSP, MaxIter, EPS, mpc, coefficients, u_data)
        else:
            _, sxb1, sxc1, LBUB1, time1 = C043().c043_CCG_n1_faster(LargeNumber, Tolerance, TimeLimitFC, TimeLimitSP, MaxIter, EPS, mpc, coefficients, u_data_train, b_display_SP)
            _, u_data, coefficients = C044().c044_reconstruction(epsilon, delta, coefficients, u_data_train_n2, sxb1, sxc1, LBUB1)
            _, _, sxb2, sxc2, LBUB2, time2, interpret = C045().c045_CCG_n2_faster(LargeNumber, Tolerance, TimeLimitFC, TimeLimitSP, MaxIter, EPS, mpc, coefficients, u_data)
        validation_cost = C046().c046_evaluate_faster(u_data_validation, coefficients, sxb2, sxc2, LBUB2)
        test_cost = C046().c046_evaluate_faster(u_data_test, coefficients, sxb2, sxc2, LBUB2)
        train_cost = C046().c046_evaluate_faster(u_data_train_original, coefficients, sxb2, sxc2, LBUB2, b_sort=False)
        train_order = np.argsort(train_cost)

        print('Calculated bound: {}'.format(LBUB2[-1, 0]))
        print('Validation bound:')
        print(validation_cost[:3])

        return validation_cost, test_cost, sxb1, sxc1, LBUB1, time1, sxb2, sxc2, LBUB2, time2, train_cost, train_order, interpret
    
    @staticmethod
    def weight2cost_P1(parameter, weight, index_u_l_predict=0, name_case='case_ieee30'):
        """
        Combine c032, c041-c043, c046
        """

        type_r = parameter['type_r']
        b_display_SP = parameter['b_display_SP']
        num_groups = parameter['num_groups']
        num_wind = parameter['num_wind']
        horizon = parameter['horizon']
        epsilon = parameter['epsilon']
        delta = parameter['delta']
        MaxIter = parameter['MaxIter']
        LargeNumber = parameter['LargeNumber']
        Tolerance = parameter['Tolerance']
        TimeLimitFC = parameter['TimeLimitFC']
        TimeLimitSP = parameter['TimeLimitSP']
        EPS = parameter['EPS']
        u_select = parameter['u_select']

        train_real, train_predict, train_n1_real, train_n1_predict, train_n2_real, train_n2_predict, validation_real, validation_predict, test_real, test_predict, error_bounds = C032().c032_calculate_weight(num_groups, num_wind, weight)
        error_mu, error_sigma, error_rho = C041().c041_initial_uncertainty(type_r, horizon, epsilon, delta, u_select, train_n1_real, train_n1_predict, train_n2_real, train_n2_predict)
        u_l_predict = validation_predict[(index_u_l_predict * horizon):((index_u_l_predict + 1) * horizon)]
        mpc, coefficients, u_data_train, u_data_train_n2, u_data_validation, u_data_test, u_data_train_original = C042().c042_dispatch_model(u_select, error_mu, error_sigma, error_rho, error_bounds, EPS, train_real, train_predict, train_n2_real, train_n2_predict, validation_real, validation_predict, test_real, test_predict, u_l_predict, name_case)

        # _, sxb1, sxc1, LBUB1, time1 = C043().c043_CCG_n1_faster(LargeNumber, Tolerance, TimeLimitFC, TimeLimitSP, MaxIter, EPS, mpc, coefficients, u_data_train, b_display_SP)
        # validation_cost = C046().c046_evaluate_faster(u_data_validation, coefficients, sxb1, sxc1, LBUB1)
        # test_cost = C046().c046_evaluate_faster(u_data_test, coefficients, sxb1, sxc1, LBUB1)
        if name_case == 'case_ieee30':
            _, sxb1, sxc1, LBUB1, time1 = C043().c043_CCG_n1(LargeNumber, Tolerance, TimeLimitFC, TimeLimitSP, MaxIter, EPS, mpc, coefficients, u_data_train, b_display_SP)
            validation_cost = C046().c046_evaluate_faster(u_data_validation, coefficients, sxb1, sxc1, LBUB1)
            test_cost = C046().c046_evaluate_faster(u_data_test, coefficients, sxb1, sxc1, LBUB1)
        else:
            _, sxb1, sxc1, LBUB1, time1 = C043().c043_CCG_n1_faster(LargeNumber, Tolerance, TimeLimitFC, TimeLimitSP, MaxIter, EPS, mpc, coefficients, u_data_train, b_display_SP)
            validation_cost = C046().c046_evaluate_faster(u_data_validation, coefficients, sxb1, sxc1, LBUB1)
            test_cost = C046().c046_evaluate_faster(u_data_test, coefficients, sxb1, sxc1, LBUB1)

        print('Calculated bound: {}'.format(LBUB1[-1, 0]))
        print('Validation bound:')
        print(validation_cost[:3])

        return validation_cost, test_cost, sxb1, sxc1, LBUB1, time1
    
    @staticmethod
    def weight2cost_RO(parameter, weight, index_u_l_predict=0, name_case='case_ieee30'):
        """
        Combine c032, c041-c043, c046
        """

        b_use_n2 = parameter['b_use_n2']
        b_display_SP = parameter['b_display_SP']
        num_groups = parameter['num_groups']
        num_wind = parameter['num_wind']
        horizon = parameter['horizon']
        epsilon = parameter['epsilon']
        delta = parameter['delta']
        MaxIter = parameter['MaxIter']
        LargeNumber = parameter['LargeNumber']
        Tolerance = parameter['Tolerance']
        TimeLimitFC = parameter['TimeLimitFC']
        TimeLimitSP = parameter['TimeLimitSP']
        EPS = parameter['EPS']
        u_select = parameter['u_select']

        train_real, train_predict, train_n1_real, train_n1_predict, train_n2_real, train_n2_predict, validation_real, validation_predict, test_real, test_predict, error_bounds = C032().c032_calculate_weight(num_groups, num_wind, weight)
        error_mu, error_sigma, error_rho = C041().c041_initial_uncertainty(b_use_n2, horizon, epsilon, delta, u_select, train_n1_real, train_n1_predict, train_n2_real, train_n2_predict)
        u_l_predict = validation_predict[(index_u_l_predict * horizon):((index_u_l_predict + 1) * horizon)]
        mpc, coefficients, u_data_train, u_data_train_n2, u_data_validation, u_data_test, u_data_train_original = C042().c042_dispatch_model(u_select, error_mu, error_sigma, error_rho, error_bounds, EPS, train_real, train_predict, train_n2_real, train_n2_predict, validation_real, validation_predict, test_real, test_predict, u_l_predict, name_case)

        # _, sxb1, sxc1, LBUB1, time1 = C043().c043_CCG_n1_faster(LargeNumber, Tolerance, TimeLimitFC, TimeLimitSP, MaxIter, EPS, mpc, coefficients, u_data_train, b_display_SP)
        # validation_cost = C046().c046_evaluate_faster(u_data_validation, coefficients, sxb1, sxc1, LBUB1)
        # test_cost = C046().c046_evaluate_faster(u_data_test, coefficients, sxb1, sxc1, LBUB1)
        if name_case == 'case_ieee30':
            _, sxb1, sxc1, LBUB1, time1 = C043().c043_CCG_n1(LargeNumber, Tolerance, TimeLimitFC, TimeLimitSP, MaxIter, EPS, mpc, coefficients, u_data_train, b_display_SP)
            validation_cost = C046().c046_evaluate_faster(u_data_validation, coefficients, sxb1, sxc1, LBUB1)
            test_cost = C046().c046_evaluate_faster(u_data_test, coefficients, sxb1, sxc1, LBUB1)
        else:
            _, sxb1, sxc1, LBUB1, time1 = C043().c043_CCG_n1_faster(LargeNumber, Tolerance, TimeLimitFC, TimeLimitSP, MaxIter, EPS, mpc, coefficients, u_data_train, b_display_SP)
            validation_cost = C046().c046_evaluate_faster(u_data_validation, coefficients, sxb1, sxc1, LBUB1)
            test_cost = C046().c046_evaluate_faster(u_data_test, coefficients, sxb1, sxc1, LBUB1)

        print('Calculated bound: {}'.format(LBUB1[-1, 0]))
        print('Validation bound:')
        print(validation_cost[:3])

        return validation_cost, test_cost, sxb1, sxc1, LBUB1, time1
    
    @staticmethod
    def weight2cost_dataRO(num_list, parameter, weight, index_u_l_predict=0, name_case='case_ieee30'):
        """
        Combine c032, c041, c042, c047, c046
        """

        type_r = parameter['type_r']
        b_display_SP = parameter['b_display_SP']
        num_groups = parameter['num_groups']
        num_wind = parameter['num_wind']
        horizon = parameter['horizon']
        epsilon = parameter['epsilon']
        delta = parameter['delta']
        MaxIter = parameter['MaxIter']
        LargeNumber = parameter['LargeNumber']
        Tolerance = parameter['Tolerance']
        TimeLimitFC = parameter['TimeLimitFC']
        TimeLimitSP = parameter['TimeLimitSP']
        EPS = parameter['EPS']
        u_select = parameter['u_select']

        train_real, train_predict, train_n1_real, train_n1_predict, train_n2_real, train_n2_predict, validation_real, validation_predict, test_real, test_predict, error_bounds = C032().c032_calculate_weight(num_groups, num_wind, weight)
        error_mu, error_sigma, error_rho = C041().c041_initial_uncertainty(type_r, horizon, epsilon, delta, u_select, train_n1_real, train_n1_predict, train_n2_real, train_n2_predict)
        u_l_predict = validation_predict[(index_u_l_predict * horizon):((index_u_l_predict + 1) * horizon)]
        mpc, coefficients, u_data_train, u_data_train_n2, u_data_validation, u_data_test, u_data_train_original = C042().c042_dispatch_model(u_select, error_mu, error_sigma, error_rho, error_bounds, EPS, train_real, train_predict, train_n2_real, train_n2_predict, validation_real, validation_predict, test_real, test_predict, u_l_predict, name_case)

        u_data_train_original = u_data_train_original[num_list, :]

        sxb1, sxc1, LBUB1, time1 = C047().c047_approx(u_data_train_original.shape[0], MaxIter, coefficients, u_data_train_original)

        validation_cost = C046().c046_evaluate_faster(u_data_validation, coefficients, sxb1, sxc1, LBUB1)
        test_cost = C046().c046_evaluate_faster(u_data_test, coefficients, sxb1, sxc1, LBUB1)

        print('Calculated bound: {}'.format(LBUB1[-1, 0]))
        print('Validation bound:')
        print(validation_cost[:3])

        return validation_cost, test_cost, sxb1, sxc1, LBUB1, time1
    
    @staticmethod
    def weight2cost_SPapprox(parameter, weight, index_u_l_predict=0, name_case='case_ieee30'):
        """
        Combine c032, c041, c042, c047, c046
        """

        type_r = parameter['type_r']
        b_display_SP = parameter['b_display_SP']
        num_groups = parameter['num_groups']
        num_wind = parameter['num_wind']
        horizon = parameter['horizon']
        epsilon = parameter['epsilon']
        delta = parameter['delta']
        MaxIter = parameter['MaxIter']
        LargeNumber = parameter['LargeNumber']
        Tolerance = parameter['Tolerance']
        TimeLimitFC = parameter['TimeLimitFC']
        TimeLimitSP = parameter['TimeLimitSP']
        EPS = parameter['EPS']
        u_select = parameter['u_select']

        train_real, train_predict, train_n1_real, train_n1_predict, train_n2_real, train_n2_predict, validation_real, validation_predict, test_real, test_predict, error_bounds = C032().c032_calculate_weight(num_groups, num_wind, weight)
        error_mu, error_sigma, error_rho = C041().c041_initial_uncertainty(type_r, horizon, epsilon, delta, u_select, train_n1_real, train_n1_predict, train_n2_real, train_n2_predict)
        u_l_predict = validation_predict[(index_u_l_predict * horizon):((index_u_l_predict + 1) * horizon)]
        mpc, coefficients, u_data_train, u_data_train_n2, u_data_validation, u_data_test, u_data_train_original = C042().c042_dispatch_model(u_select, error_mu, error_sigma, error_rho, error_bounds, EPS, train_real, train_predict, train_n2_real, train_n2_predict, validation_real, validation_predict, test_real, test_predict, u_l_predict, name_case)

        sxb1, sxc1, LBUB1, time1 = C047().c047_approx(np.ceil((1 - epsilon) * u_data_train_original.shape[0]).astype(int), MaxIter, coefficients, u_data_train_original)

        validation_cost = C046().c046_evaluate_faster(u_data_validation, coefficients, sxb1, sxc1, LBUB1)
        test_cost = C046().c046_evaluate_faster(u_data_test, coefficients, sxb1, sxc1, LBUB1)

        print('Calculated bound: {}'.format(LBUB1[-1, 0]))
        print('Validation bound:')
        print(validation_cost[:3])

        return validation_cost, test_cost, sxb1, sxc1, LBUB1, time1
    
    @staticmethod
    def weight2cost_SP(parameter, weight, index_u_l_predict=0, name_case='case_ieee30'):
        """
        Combine c032, c041, c042, c047, c046
        """

        type_r = parameter['type_r']
        b_display_SP = parameter['b_display_SP']
        num_groups = parameter['num_groups']
        num_wind = parameter['num_wind']
        horizon = parameter['horizon']
        epsilon = parameter['epsilon']
        delta = parameter['delta']
        MaxIter = parameter['MaxIter']
        LargeNumber = parameter['LargeNumber']
        Tolerance = parameter['Tolerance']
        TimeLimitFC = parameter['TimeLimitFC']
        TimeLimitSP = parameter['TimeLimitSP']
        EPS = parameter['EPS']
        u_select = parameter['u_select']

        train_real, train_predict, train_n1_real, train_n1_predict, train_n2_real, train_n2_predict, validation_real, validation_predict, test_real, test_predict, error_bounds = C032().c032_calculate_weight(num_groups, num_wind, weight)
        error_mu, error_sigma, error_rho = C041().c041_initial_uncertainty(type_r, horizon, epsilon, delta, u_select, train_n1_real, train_n1_predict, train_n2_real, train_n2_predict)
        u_l_predict = validation_predict[(index_u_l_predict * horizon):((index_u_l_predict + 1) * horizon)]
        mpc, coefficients, u_data_train, u_data_train_n2, u_data_validation, u_data_test, u_data_train_original = C042().c042_dispatch_model(u_select, error_mu, error_sigma, error_rho, error_bounds, EPS, train_real, train_predict, train_n2_real, train_n2_predict, validation_real, validation_predict, test_real, test_predict, u_l_predict, name_case)

        sxb1, sxc1, LBUB1, time1 = C047().c047_MILP(epsilon, LargeNumber, MaxIter, coefficients, u_data_train_original)
        validation_cost = C046().c046_evaluate_faster(u_data_validation, coefficients, sxb1, sxc1, LBUB1)
        test_cost = C046().c046_evaluate_faster(u_data_test, coefficients, sxb1, sxc1, LBUB1)

        print('Calculated bound: {}'.format(LBUB1[-1, 0]))
        print('Validation bound:')
        print(validation_cost[:3])

        return validation_cost, test_cost, sxb1, sxc1, LBUB1, time1
    