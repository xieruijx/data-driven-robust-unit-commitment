import numpy as np

from utils.c032_weight_calculation import C032
from utils.c041_uncertainty_initiation import C041
from utils.c042_dispatch_model import C042
from utils.c043_CCG_ellipsoid import C043
from utils.c044_reconstruction import C044
from utils.c045_CCG_polyhedron import C045
from utils.c046_evaluation import C046
from utils.c047_SP_enumeration import C047
from utils.case import Case

class Optimization(object):
    """
    Optimization class for the optimization process
    """
    
    @staticmethod
    def weight2cost(parameter, weight, type_r='n1', type_SP=None, index_u_l_predict=0, name_case='case_ieee30', type_u_l='test'):
        """
        Combine c032, c041-c046
        """

        b_faster = parameter['b_faster']
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

        if type_u_l == 'train':
            u_l_predict = train_predict[(index_u_l_predict * horizon):((index_u_l_predict + 1) * horizon)]
        elif type_u_l == 'validation':
            u_l_predict = validation_predict[(index_u_l_predict * horizon):((index_u_l_predict + 1) * horizon)]
        elif type_u_l == 'test':
            u_l_predict = test_predict[(index_u_l_predict * horizon):((index_u_l_predict + 1) * horizon)]
        else:
            raise RuntimeError('The type of u_l (type_u_l) is wrong')

        error_mu, error_sigma, error_rho = C041().c041_initial_uncertainty(type_r, horizon, epsilon, delta, u_select, train_n1_real, train_n1_predict, train_n2_real, train_n2_predict)

        mpc, coefficients, u_data_train, u_data_train_n2, u_data_validation, u_data_test, u_data_train_original = C042().c042_dispatch_model(u_select, error_mu, error_sigma, error_rho, error_bounds, EPS, train_real, train_predict, train_n2_real, train_n2_predict, validation_real, validation_predict, test_real, test_predict, u_l_predict, name_case)

        if type_SP == None: # RO methods for ellipsoidal or polyhedral uncertainty set
            if not b_faster:
                _, sxb1, sxc1, LBUB1, time1, interpret = C043().c043_CCG_n1(LargeNumber, Tolerance, TimeLimitFC, TimeLimitSP, MaxIter, EPS, mpc, coefficients, u_data_train, b_display_SP)
            else:
                _, sxb1, sxc1, LBUB1, time1, interpret = C043().c043_CCG_n1_faster(LargeNumber, Tolerance, TimeLimitFC, TimeLimitSP, MaxIter, EPS, mpc, coefficients, u_data_train, b_display_SP)

            if type_r == 'n1': # Include reconstruction and CCG for polyhedral uncertainty set
                _, u_data, coefficients, _ = C044().c044_reconstruction(epsilon, delta, coefficients, u_data_train_n2, sxb1, sxc1, LBUB1)
                if not b_faster:
                    _, _, sxb2, sxc2, LBUB2, time2, interpret = C045().c045_CCG_n2(LargeNumber, Tolerance, TimeLimitFC, TimeLimitSP, MaxIter, EPS, mpc, coefficients, u_data)
                else:
                    _, _, sxb2, sxc2, LBUB2, time2, interpret = C045().c045_CCG_n2_faster(LargeNumber, Tolerance, TimeLimitFC, TimeLimitSP, MaxIter, EPS, mpc, coefficients, u_data)
            else: # type_r = 'n2', 'n_m', 'n_q'
                sxb2 = sxb1
                sxc2 = sxc1
                LBUB2 = LBUB1
                time2 = 0
        else: # SP/RO methods for discrete uncertainty sets
            if type_SP == 'MILP':
                sxb1, sxc1, LBUB1, time1, interpret = C047().c047_MILP(epsilon, mpc, LargeNumber, MaxIter, coefficients, u_data_train_original)
            elif type_SP == 'approx':
                sxb1, sxc1, LBUB1, time1, interpret = C047().c047_approx(epsilon, mpc, MaxIter, coefficients, u_data_train_original)
            else:
                raise RuntimeError('The type of SP (type_SP) is wrong')
            sxb2 = sxb1
            sxc2 = sxc1
            LBUB2 = LBUB1
            time2 = 0

        validation_cost = C046().c046_evaluate_faster(u_data_validation, coefficients, sxb2, sxc2, LBUB2, b_sort=False)
        test_cost = C046().c046_evaluate_faster(u_data_test, coefficients, sxb2, sxc2, LBUB2, b_sort=False)
        train_cost = C046().c046_evaluate_faster(u_data_train_original, coefficients, sxb2, sxc2, LBUB2, b_sort=False)
        train_order = np.argsort(train_cost)

        print('Calculated bound: {}'.format(LBUB2[-1, 0]))
        print('Validation bound:')
        print(validation_cost[:3])
        time = [time1, time2]

        return validation_cost, test_cost, sxb1, sxc1, LBUB1, sxb2, sxc2, LBUB2, time, train_cost, train_order, interpret
    
    @staticmethod
    def weight2ellipsoid(parameter, weight, type_r='n1', index_u_l_predict=0, name_case='case_ieee30', type_u_l='test'):
        """
        Combine c032 and c041 and output the parameters of ellipsoidal uncertainty set
        """

        num_groups = parameter['num_groups']
        num_wind = parameter['num_wind']
        horizon = parameter['horizon']
        epsilon = parameter['epsilon']
        delta = parameter['delta']
        u_select = parameter['u_select']

        _, train_predict, train_n1_real, train_n1_predict, train_n2_real, train_n2_predict, _, validation_predict, _, test_predict, error_bounds = C032().c032_calculate_weight(num_groups, num_wind, weight)

        if type_u_l == 'train':
            u_l_predict = train_predict[(index_u_l_predict * horizon):((index_u_l_predict + 1) * horizon)]
        elif type_u_l == 'validation':
            u_l_predict = validation_predict[(index_u_l_predict * horizon):((index_u_l_predict + 1) * horizon)]
        elif type_u_l == 'test':
            u_l_predict = test_predict[(index_u_l_predict * horizon):((index_u_l_predict + 1) * horizon)]
        else:
            raise RuntimeError('The type of u_l (type_u_l) is wrong')

        error_mu, error_sigma, error_rho = C041().c041_initial_uncertainty(type_r, horizon, epsilon, delta, u_select, train_n1_real, train_n1_predict, train_n2_real, train_n2_predict)

        parameter['u_l_predict'] = u_l_predict
        parameter['error_mu'] = error_mu
        parameter['error_sigma'] = error_sigma
        parameter['error_rho'] = error_rho
        parameter['error_bounds'] = error_bounds

        ## Load case
        if name_case == 'case_ieee30':
            mpc = Case().case_ieee30_modified(parameter)
        elif name_case == 'case118':
            mpc = Case().case118_modified(parameter)
        else:
            raise RuntimeError('The case name is wrong')
        
        mpc = Case().process_case(mpc)
        u_l_predict = mpc['u_l_predict']
        error_lb = mpc['error_lb']
        error_ub = mpc['error_ub']
        u_lu = mpc['u_lu']
        u_ll = mpc['u_ll']

        return error_mu, error_sigma, error_rho, u_l_predict, error_lb, error_ub, u_lu, u_ll
    
    @staticmethod
    def weight2polyhedron(parameter, weight, index_u_l_predict=0, name_case='case_ieee30', type_u_l='test'):
        """
        Combine c032 and c041-C044
        """
        b_faster = parameter['b_faster']
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

        if type_u_l == 'train':
            u_l_predict = train_predict[(index_u_l_predict * horizon):((index_u_l_predict + 1) * horizon)]
        elif type_u_l == 'validation':
            u_l_predict = validation_predict[(index_u_l_predict * horizon):((index_u_l_predict + 1) * horizon)]
        elif type_u_l == 'test':
            u_l_predict = test_predict[(index_u_l_predict * horizon):((index_u_l_predict + 1) * horizon)]
        else:
            raise RuntimeError('The type of u_l (type_u_l) is wrong')

        error_mu, error_sigma, error_rho = C041().c041_initial_uncertainty('n1', horizon, epsilon, delta, u_select, train_n1_real, train_n1_predict, train_n2_real, train_n2_predict)

        mpc, coefficients, u_data_train, u_data_train_n2, _, _, _ = C042().c042_dispatch_model(u_select, error_mu, error_sigma, error_rho, error_bounds, EPS, train_real, train_predict, train_n2_real, train_n2_predict, validation_real, validation_predict, test_real, test_predict, u_l_predict, name_case)

        if not b_faster:
            _, sxb1, sxc1, LBUB1, _, _ = C043().c043_CCG_n1(LargeNumber, Tolerance, TimeLimitFC, TimeLimitSP, MaxIter, EPS, mpc, coefficients, u_data_train, b_display_SP)
        else:
            _, sxb1, sxc1, LBUB1, _, _ = C043().c043_CCG_n1_faster(LargeNumber, Tolerance, TimeLimitFC, TimeLimitSP, MaxIter, EPS, mpc, coefficients, u_data_train, b_display_SP)

        _, _, coefficients, u_data_outside = C044().c044_reconstruction(epsilon, delta, coefficients, u_data_train_n2, sxb1, sxc1, LBUB1)

        return coefficients, u_data_outside