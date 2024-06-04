import numpy as np
import pandas as pd
import time
import gurobipy as gp
from gurobipy import GRB

from utils.combhandler import CombHandler
from utils.case import Case

class Optimization(object):
    """
    Optimization class for the optimization process
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

        optimization = Optimization()

        ## Combine prediction and calculate MSE
        df_train = optimization.combine_predict('train', num_groups, weight)
        df_train_n1 = optimization.combine_predict('train_n1', num_groups, weight)
        df_train_n2 = optimization.combine_predict('train_n2', num_groups, weight)
        df_validation = optimization.combine_predict('validation', num_groups, weight)
        df_test = optimization.combine_predict('test', num_groups, weight)

        ## Transform dataframes into matrices. Columns are load0-load20, wind1, wind2. Rows are periods.
        list_name = ['load' + str(i) for i in range(num_groups)]
        list_name.extend(['wind1', 'wind2'])
        train_real, train_predict = optimization.df2matrix(df_train, list_name)
        train_n1_real, train_n1_predict = optimization.df2matrix(df_train_n1, list_name)
        train_n2_real, train_n2_predict = optimization.df2matrix(df_train_n2, list_name)
        validation_real, validation_predict = optimization.df2matrix(df_validation, list_name)
        test_real, test_predict = optimization.df2matrix(df_test, list_name)

        ## Calculate error bounds
        list_real = [train_real, validation_real, test_real]
        list_predict = [train_predict, validation_predict, test_predict]
        error_bounds = optimization.error_bounds(list_real, list_predict)

        return train_real, train_predict, train_n1_real, train_n1_predict, train_n2_real, train_n2_predict, validation_real, validation_predict, test_real, test_predict, error_bounds

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

    @staticmethod
    def calculate_u_data(real, predict, mpc, EPS, b_ellipsoid):
        """
        Calculate the uncertainty data from real and predict data
        Revise it into the uncertainty set according to requirements
        """
        case = Case()

        real = real[:, mpc['u_select']]
        predict = predict[:, mpc['u_select']]

        num_data = real.shape[0] // mpc['n_t']
        u_data = np.zeros((num_data, mpc['n_t'] * mpc['n_u']))
        for i in range(num_data):
            u_data[i, :] = mpc['u_l_predict'] - predict[(i * mpc['n_t']):((i + 1) * mpc['n_t'])].reshape((-1,)) + real[(i * mpc['n_t']):((i + 1) * mpc['n_t'])].reshape((-1,))

        # b_ellipsoid = True: Revise the point into the ellipsoid uncertainty set if it is not
        # b_ellipsoid = False: Revise the point to satisfy the bounds if it is not
        for i in range(num_data):
            if not case.test_u(u_data[i, :], mpc, b_print=True, b_ellipsoid=b_ellipsoid):
                print('Data {} is not in the uncertainty set.'.format(i))
                u_data[i, :] = case.revise_u(u_data[i, :], mpc, EPS, b_print=True, b_ellipsoid=b_ellipsoid)

        # Test whether the points are in the ellipsoid uncertainty set
        num_not_in_uncertainty_set = 0
        for i in range(num_data):
            if not case.test_u(u_data[i, :], mpc, b_print=True, b_ellipsoid=True):
                print('Data {} is not in the ellipsoid uncertainty set.'.format(i))
                num_not_in_uncertainty_set += 1
        print('{} out of {} data are not in the ellipsoid uncertainty set.'.format(num_not_in_uncertainty_set, num_data))
        
        return u_data
    
    @staticmethod
    def c042_dispatch_model(u_select, error_mu, error_sigma, error_rho, error_bounds, EPS, train_real, train_predict, train_n2_real, train_n2_predict, validation_real, validation_predict, test_real, test_predict, u_l_predict, name_case):
        """
        Generate dispatch model and the coefficients of the optimization problem
        """
        print('(((((((((((((((((((((((((((((c042)))))))))))))))))))))))))))))')
        case = Case()
        optimization = Optimization()
        
        ## Set the parameters of the case
        parameter = {}
        parameter['u_select'] = u_select
        parameter['u_l_predict'] = u_l_predict
        parameter['error_mu'] = error_mu
        parameter['error_sigma'] = error_sigma
        parameter['error_rho'] = error_rho
        parameter['error_bounds'] = error_bounds

        ## Load case
        if name_case == 'case_ieee30':
            mpc = case.case_ieee30_modified(parameter)
        elif name_case == 'case118':
            mpc = case.case118_modified(parameter)
        else:
            mpc = case.case_ieee30_modified(parameter)

        mpc = case.process_case(mpc)

        print('Construct uncertainty data from train (ellipsoid)')
        u_data_train = optimization.calculate_u_data(train_real, train_predict, mpc, EPS, b_ellipsoid=True)
        u_data_train_original = optimization.calculate_u_data(train_real, train_predict, mpc, EPS, b_ellipsoid=False)

        print('Construct uncertainty data from train n2 (bound)')
        u_data_train_n2 = optimization.calculate_u_data(train_n2_real, train_n2_predict, mpc, EPS, b_ellipsoid=False)

        print('Construct uncertainty data from validation (bound)')
        u_data_validation = optimization.calculate_u_data(validation_real, validation_predict, mpc, EPS, b_ellipsoid=False)

        print('Construct uncertainty data from test (bound)')
        u_data_test = optimization.calculate_u_data(test_real, test_predict, mpc, EPS, b_ellipsoid=False)

        u_l_predict = mpc['u_l_predict'].reshape((mpc['n_t'], mpc['n_u']))

        ## Define variables and get model
        try:
            # Create a new model
            model = gp.Model('Modeling')

            # Create variables
            num_var = np.zeros((4,)) # Uncertain, day-ahead binary, day-ahead continuous, real-time

            u_l = model.addMVar((mpc['n_t'], mpc['n_u']), lb=-float('inf'), vtype=GRB.CONTINUOUS) # Uncertain load demand
            model.update()
            num_var[0] = model.NumVars - 0
            
            x_og = model.addMVar((mpc['n_t'], mpc['n_g']), vtype=GRB.BINARY) # Day-ahead generator on
            x_ou = model.addMVar((mpc['n_t'], mpc['n_g']), vtype=GRB.BINARY) # Day-ahead generator up
            x_od = model.addMVar((mpc['n_t'], mpc['n_g']), vtype=GRB.BINARY) # Day-ahead generator down
            model.update()
            num_var[1] = model.NumVars - 0

            x_pg = model.addMVar((mpc['n_t'], mpc['n_g']), lb=-float('inf'), vtype=GRB.CONTINUOUS) # Day-ahead active power generation
            x_rp = model.addMVar((mpc['n_t'], mpc['n_g']), lb=-float('inf'), vtype=GRB.CONTINUOUS) # Day-ahead upper power reserve
            x_rn = model.addMVar((mpc['n_t'], mpc['n_g']), lb=-float('inf'), vtype=GRB.CONTINUOUS) # Day-ahead down power reserve
            model.update()
            num_var[2] = model.NumVars - 0
            
            y_rp = model.addMVar((mpc['n_t'], mpc['n_g']), lb=-float('inf'), vtype=GRB.CONTINUOUS) # Real-time upper power
            y_rn = model.addMVar((mpc['n_t'], mpc['n_g']), lb=-float('inf'), vtype=GRB.CONTINUOUS) # Real-time down power
            model.update()
            num_var[3] = model.NumVars - 0

            # Set objective
            o_x_pg = gp.quicksum(mpc['c_x_pg'][g] * x_pg[t, g] for t in range(mpc['n_t']) for g in range(mpc['n_g']))
            o_x_rp = gp.quicksum(mpc['c_x_rp'][g] * x_rp[t, g] for t in range(mpc['n_t']) for g in range(mpc['n_g']))
            o_x_rn = gp.quicksum(mpc['c_x_rn'][g] * x_rn[t, g] for t in range(mpc['n_t']) for g in range(mpc['n_g']))
            o_x_og = gp.quicksum(mpc['c_x_og'][g] * x_og[t, g] for t in range(mpc['n_t']) for g in range(mpc['n_g']))
            o_x_ou = gp.quicksum(mpc['c_x_ou'][g] * x_ou[t, g] for t in range(mpc['n_t']) for g in range(mpc['n_g']))
            o_x_od = gp.quicksum(mpc['c_x_od'][g] * x_od[t, g] for t in range(mpc['n_t']) for g in range(mpc['n_g']))
            o_x = gp.quicksum([o_x_pg, o_x_rp, o_x_rn, o_x_og, o_x_ou, o_x_od])
            o_y_rp = gp.quicksum(mpc['c_y_rp'][g] * y_rp[t, g] for t in range(mpc['n_t']) for g in range(mpc['n_g']))
            o_y_rn = gp.quicksum(mpc['c_y_rn'][g] * y_rn[t, g] for t in range(mpc['n_t']) for g in range(mpc['n_g']))
            o_y = gp.quicksum([o_y_rp, o_y_rn])
            model.setObjective(gp.quicksum([o_x, o_y]), GRB.MINIMIZE)

            # Add constraints
            num_con = np.zeros((4,)) # Day-ahead =, day-ahead >, real-time =, real-time >

            # Day-ahead
            # Power flow 1
            model.addConstrs((x_pg[t, :] @ np.ones((mpc['n_g'],)) == mpc['PD'][t, :] @ np.ones((mpc['n_b'],)) + u_l_predict[t, :] @ mpc['bus_u'].T @ np.ones((mpc['n_b'],)) for t in range(mpc['n_t'])), name='d_pf_e')
            # Unit commitment 1
            model.addConstr(x_og[1:, :] - x_og[:-1, :] == x_ou[1:, :] - x_od[1:, :], name='d_UC_logical')
            model.update()
            num_con[0] = model.NumConstrs - 0

            # Power flow 2
            model.addConstrs((mpc['PTDF'][l, :] @ (x_pg[t, :] @ mpc['bus_gen'].T - mpc['PD'][t, :] - u_l_predict[t, :] @ mpc['bus_u'].T) >= - mpc['S'][l] for t in range(mpc['n_t']) for l in range(mpc['n_l'])), name='d_pf_l_lb')
            model.addConstrs((mpc['S'][l] >= mpc['PTDF'][l, :] @ (x_pg[t, :] @ mpc['bus_gen'].T - mpc['PD'][t, :] - u_l_predict[t, :] @ mpc['bus_u'].T) for t in range(mpc['n_t']) for l in range(mpc['n_l'])), name='d_pf_l_ub')
            # Generator dispatch
            model.addConstrs((x_rp[t, :] >= 0 for t in range(mpc['n_t'])), name='d_ED_rp_lb')
            model.addConstrs((mpc['Ramp'] * x_og[t, :] >= x_rp[t, :] for t in range(mpc['n_t'])), name='d_ED_rp_ub')
            model.addConstrs((x_rn[t, :] >= 0 for t in range(mpc['n_t'])), name='d_ED_rn_lb')
            model.addConstrs((mpc['Ramp'] * x_og[t, :] >= x_rn[t, :] for t in range(mpc['n_t'])), name='d_ED_rn_ub')
            model.addConstrs((x_pg[t, :] >= mpc['Pmin'] * x_og[t, :] + x_rn[t, :] for t in range(mpc['n_t'])), name='d_ED_pg_lb')
            model.addConstrs((mpc['Pmax'] * x_og[t, :] - x_rp[t, :] >= x_pg[t, :] for t in range(mpc['n_t'])), name='d_ED_pg_ub')
            model.addConstrs((mpc['Ramp'] * x_og[t, :] + mpc['Pmax'] * x_ou[t + 1, :] >= x_pg[t + 1, :] + x_rp[t + 1, :] - x_pg[t, :] + x_rn[t, :] for t in range(mpc['n_t'] - 1)), name='d_ED_ramp_u')
            model.addConstrs((x_pg[t + 1, :] - x_rn[t + 1, :] - x_pg[t, :] - x_rp[t, :] >= - mpc['Ramp'] * x_og[t + 1, :] - mpc['Pmax'] * x_od[t + 1, :] for t in range(mpc['n_t'] - 1)), name='d_ED_ramp_d')
            # Unit commitment 2
            model.addConstrs((gp.quicksum(x_og[t:(t + mpc['UTDT']), g]) >= mpc['UTDT'] * x_ou[t, g] for t in range(mpc['n_t'] - mpc['UTDT'] + 1) for g in range(mpc['n_g'])), name='d_UC_ou1')
            model.addConstrs((gp.quicksum(x_og[t:, g]) >= (mpc['n_t'] - t) * x_ou[t, g] for t in range(mpc['n_t'] - mpc['UTDT'] + 1, mpc['n_t']) for g in range(mpc['n_g'])), name='d_UC_ou2')
            model.addConstrs((mpc['UTDT'] - gp.quicksum(x_og[t:(t + mpc['UTDT']), g]) >= mpc['UTDT'] * x_od[t, g] for t in range(mpc['n_t'] - mpc['UTDT'] + 1) for g in range(mpc['n_g'])), name='d_UC_od1')
            model.addConstrs((mpc['n_t'] - t - gp.quicksum(x_og[t:, g]) >= (mpc['n_t'] - t) * x_od[t, g] for t in range(mpc['n_t'] - mpc['UTDT'] + 1, mpc['n_t']) for g in range(mpc['n_g'])), name='d_UC_od2')
            model.addConstrs((1 >= x_ou[t, g] + x_od[t, g] for t in range(mpc['n_t']) for g in range(mpc['n_g'])), name='d_UC_complementary')
            model.update()
            num_con[1] = model.NumConstrs - 0

            # Real-time
            # Power flow
            model.addConstrs(((x_pg[t, :] + y_rp[t, :] - y_rn[t, :]) @ np.ones((mpc['n_g'],)) == mpc['PD'][t, :] @ np.ones((mpc['n_b'],)) + u_l[t, :] @ mpc['bus_u'].T @ np.ones((mpc['n_b'],)) for t in range(mpc['n_t'])), name='r_pf_e') # the only slack
            model.update()
            num_con[2] = model.NumConstrs - 0

            model.addConstrs((mpc['PTDF'][l, :] @ ((x_pg[t, :] + y_rp[t, :] - y_rn[t, :]) @ mpc['bus_gen'].T - mpc['PD'][t, :] - u_l[t, :] @ mpc['bus_u'].T) >= - mpc['S'][l] for t in range(mpc['n_t']) for l in range(mpc['n_l'])), name='r_pf_l_lb')
            model.addConstrs((mpc['S'][l] >= mpc['PTDF'][l, :] @ ((x_pg[t, :] + y_rp[t, :] - y_rn[t, :]) @ mpc['bus_gen'].T - mpc['PD'][t, :] - u_l[t, :] @ mpc['bus_u'].T) for t in range(mpc['n_t']) for l in range(mpc['n_l'])), name='r_pf_l_ub')
            # Generator dispatch
            model.addConstrs((x_rp[t, :] >= y_rp[t, :] for t in range(mpc['n_t'])), name='r_ED_rp_ub')
            model.addConstrs((x_rn[t, :] >= y_rn[t, :] for t in range(mpc['n_t'])), name='r_ED_rn_ub')
            model.addConstrs((y_rp[t, :] >= 0 for t in range(mpc['n_t'])), name='r_ED_rp_lb')
            model.addConstrs((y_rn[t, :] >= 0 for t in range(mpc['n_t'])), name='r_ED_rn_lb')
            model.update()
            num_con[3] = model.NumConstrs - 0

            ## Output model coefficients
            A = model.getA()
            B = np.array(model.getAttr('RHS', model.getConstrs()))
            C = np.array(model.getAttr('Obj',model.getVars()))
            sense = model.getAttr('Sense', model.getConstrs())
            for i, x in enumerate(sense):
                if x == '<':
                    A[i, :] = - A[i, :]
                    B[i] = - B[i]
            
            # Day-ahead equation: Adexb * xb + Adexc * xc = Bde
            Adexb = A[:int(num_con[0]), int(num_var[0]):int(num_var[1])]
            Adexc = A[:int(num_con[0]), int(num_var[1]):int(num_var[2])]

            # Day-ahead inequality: Adixb * xb + Adixc * xc >= Bdi
            Adixb = A[int(num_con[0]):int(num_con[1]), int(num_var[0]):int(num_var[1])]
            Adixc = A[int(num_con[0]):int(num_con[1]), int(num_var[1]):int(num_var[2])]

            # Real-time equation: Areu * u + Arexc * xc + Arey * y = Bre
            Areu = A[int(num_con[1]):int(num_con[2]), :int(num_var[0])]
            Arexc = A[int(num_con[1]):int(num_con[2]), int(num_var[1]):int(num_var[2])]
            Arey = A[int(num_con[1]):int(num_con[2]), int(num_var[2]):int(num_var[3])]

            # Real-time inequality: Arixc * xc + Ariy * y >= Bri
            Arixc = A[int(num_con[2]):int(num_con[3]), int(num_var[1]):int(num_var[2])]
            Ariy = A[int(num_con[2]):int(num_con[3]), int(num_var[2]):int(num_var[3])]

            # Right-hand side
            Bde = B[:int(num_con[0])]
            Bdi = B[int(num_con[0]):int(num_con[1])]
            Bre = B[int(num_con[1]):int(num_con[2])]
            Bri = B[int(num_con[2]):int(num_con[3])]

            # Objective
            Cdxb = C[int(num_var[0]):int(num_var[1])]
            Cdxc = C[int(num_var[1]):int(num_var[2])]
            Cry = C[int(num_var[2]):int(num_var[3])]
            
        except gp.GurobiError as e:
            print(f"Error code {e.errno}: {e}")

        except AttributeError:
            print("Encountered an attribute error")

        coefficients = {}
        coefficients['Adexb'] = Adexb
        coefficients['Adexc'] = Adexc
        coefficients['Adixb'] = Adixb
        coefficients['Adixc'] = Adixc
        coefficients['Areu'] = Areu
        coefficients['Arexc'] = Arexc
        coefficients['Arey'] = Arey
        coefficients['Arixc'] = Arixc
        coefficients['Ariy'] = Ariy
        coefficients['Bde'] = Bde
        coefficients['Bdi'] = Bdi
        coefficients['Bre'] = Bre
        coefficients['Bri'] = Bri
        coefficients['Cdxb'] = Cdxb
        coefficients['Cdxc'] = Cdxc
        coefficients['Cry'] = Cry

        return mpc, coefficients, u_data_train, u_data_train_n2, u_data_validation, u_data_test, u_data_train_original
    
    @staticmethod
    def c043_CCG_n1(LargeNumber, Tolerance, TimeLimitFC, TimeLimitSP, MaxIter, EPS, mpc, coefficients, u_data_train, b_display_SP):
        """
        CCG with the ellipsoid uncertainty set
        """
        ## Time
        time_start = time.time()

        ## Load case
        case = Case()

        ## Load compact form
        Adexb = coefficients['Adexb']
        Adexc = coefficients['Adexc']
        Adixb = coefficients['Adixb']
        Adixc = coefficients['Adixc']
        Areu = coefficients['Areu']
        Arexc = coefficients['Arexc']
        Arey = coefficients['Arey']
        Arixc = coefficients['Arixc']
        Ariy = coefficients['Ariy']
        Bde = coefficients['Bde']
        Bdi = coefficients['Bdi']
        Bre = coefficients['Bre']
        Bri = coefficients['Bri']
        Cdxb = coefficients['Cdxb']
        Cdxc = coefficients['Cdxc']
        Cry = coefficients['Cry']

        ## Load u_data_train
        u_data = u_data_train
        num_data = u_data.shape[0]

        try:
            ## Initiation
            # MP: Master problem
            MP = gp.Model('Master')
            xbMP = MP.addMVar((Cdxb.shape[0],), vtype=GRB.BINARY)
            xcMP = MP.addMVar((Cdxc.shape[0],), lb=-float('inf'), vtype=GRB.CONTINUOUS)
            yMP = MP.addMVar((MaxIter, Cry.shape[0]), lb=-float('inf'), vtype=GRB.CONTINUOUS)
            yMPdata = MP.addMVar((num_data, Cry.shape[0]), lb=-float('inf'), vtype=GRB.CONTINUOUS) # y for uncertainty data
            zetaMP = MP.addVar(lb=-LargeNumber, vtype=GRB.CONTINUOUS)
            MP.setObjective(Cdxb @ xbMP + Cdxc @ xcMP + zetaMP, GRB.MINIMIZE)
            MP.addConstr(Adexb @ xbMP + Adexc @ xcMP == Bde, name='de')
            MP.addConstr(Adixb @ xbMP + Adixc @ xcMP >= Bdi, name='di')
            for i in range(num_data): # Add uncertainty data as initiation
                MP.addConstr(zetaMP >= Cry @ yMPdata[i, :], name='rcd')
                MP.addConstr(Areu @ u_data[i, :] + Arexc @ xcMP + Arey @ yMPdata[i, :] == Bre, name='red')
                MP.addConstr(Arixc @ xcMP + Ariy @ yMPdata[i, :] >= Bri, name='rid')
            MP.setParam('OutputFlag', 0) 

            # Feasibility check problem by mountain climbing
            # FC1: Fix uncertainty and optimize dual variables
            FC1 = gp.Model('Feasibility-Dual')
            muFC1 = FC1.addMVar((Bre.shape[0],), lb=-float('inf'), vtype=GRB.CONTINUOUS) # Dual variable of equation
            etaFC1 = FC1.addMVar((Bri.shape[0],), lb=-float('inf'), vtype=GRB.CONTINUOUS) # Dual variable of inequality
            FC1.addConstr(Arey.T @ muFC1 + Ariy.T @ etaFC1 == 0, name='de')
            FC1.addConstr(muFC1 <= 1, name='dru')
            FC1.addConstr(muFC1 >= -1, name='drl')
            FC1.addConstr(etaFC1 >= 0, name='di')
            FC1.setParam('OutputFlag', 0) 
            # FC2: Fix dual variables and optimize uncertainty
            FC2 = gp.Model('Feasibility-Uncertainty')
            uFC2 = FC2.addMVar((Areu.shape[1],), lb=-float('inf'), vtype=GRB.CONTINUOUS) # real u
            u_ldFC2 = FC2.addMVar((Areu.shape[1],), lb=-float('inf'), vtype=GRB.CONTINUOUS) # error - mu
            FC2.addConstr(u_ldFC2 == mpc['u_l_predict'] - uFC2 - mpc['error_mu'], name='u_e')
            FC2.addConstr(uFC2 >= mpc['u_ll'], name='u_lb')
            FC2.addConstr(mpc['u_lu'] >= uFC2, name='u_ub')
            FC2.addConstr(mpc['u_l_predict'] - uFC2 >= mpc['error_lb'], name='u_e_lb')
            FC2.addConstr(mpc['error_ub'] >= mpc['u_l_predict'] - uFC2, name='u_e_ub')
            FC2.addMQConstr(Q=mpc['error_sigma_inv'], c=None, sense='<', rhs=mpc['error_rho'], xQ_L=u_ldFC2, xQ_R=u_ldFC2, name='u_q')
            FC2.setParam('OutputFlag', 0) 

            ## Subproblem by mountain climbing
            # SP1: Fix uncertainty and optimize dual variables
            SP1 = gp.Model('Subproblem-Dual')
            muSP1 = SP1.addMVar((Bre.shape[0],), lb=-float('inf'), vtype=GRB.CONTINUOUS) # Dual variable of equation
            etaSP1 = SP1.addMVar((Bri.shape[0],), lb=-float('inf'), vtype=GRB.CONTINUOUS) # Dual variable of inequality
            SP1.addConstr(Arey.T @ muSP1 + Ariy.T @ etaSP1 == Cry, name='de')
            SP1.addConstr(etaSP1 >= 0, name='di')
            SP1.setParam('OutputFlag', 0) 
            # SP2: Fix dual variables and optimize uncertainty
            SP2 = gp.Model('Subproblem-Uncertainty')
            uSP2 = SP2.addMVar((Areu.shape[1],), lb=-float('inf'), vtype=GRB.CONTINUOUS)
            u_ldSP2 = SP2.addMVar((Areu.shape[1],), lb=-float('inf'), vtype=GRB.CONTINUOUS) # error - mu
            SP2.addConstr(u_ldSP2 == mpc['u_l_predict'] - uSP2 - mpc['error_mu'], name='u_e')
            SP2.addConstr(uSP2 >= mpc['u_ll'], name='u_lb')
            SP2.addConstr(mpc['u_lu'] >= uSP2, name='u_ub')
            SP2.addConstr(mpc['u_l_predict'] - uSP2 >= mpc['error_lb'], name='u_e_lb')
            SP2.addConstr(mpc['error_ub'] >= mpc['u_l_predict'] - uSP2, name='u_e_ub')
            SP2.addMQConstr(Q=mpc['error_sigma_inv'], c=None, sense='<', rhs=mpc['error_rho'], xQ_L=u_ldSP2, xQ_R=u_ldSP2, name='u_q')
            SP2.setParam('OutputFlag', 0) 

            # Other initiation
            LB = -float('inf') * np.ones((MaxIter,))
            UBU = float('inf') * np.ones((MaxIter,)) # Theoretical upper bound
            UBL = float('inf') * np.ones((MaxIter,)) # Founded upper bound
            ulist = [] # The list of worst-case uncertainty
            Iter = 0

            ## Iteration
            while True:
                print('**************************************************************')
                print('Begin iteration: {}'.format(Iter))

                print('******************************MP******************************')
                MP.optimize()
                sxb = xbMP.X
                sxc = xcMP.X
                LB[Iter] = MP.ObjVal
                MPObjX = Cdxb @ sxb + Cdxc @ sxc

                print('LB: {} >= first-stage cost: {}'.format(LB[Iter], MPObjX))

                if (Iter > 0) & (LB[Iter] < LB[Iter - 1] * (1 + EPS)):
                    print('Cannot improve. Terminate.')
                    break

                # FC: Mountain climbing as initiation
                su = mpc['u_l_predict'] - mpc['error_mu'] # Initialize u
                for i in range(3):
                    FC1.setObjective((Bre - Areu @ su - Arexc @ sxc) @ muFC1 + (Bri - Arixc @ sxc) @ etaFC1, GRB.MAXIMIZE)
                    FC1.optimize()
                    smu = muFC1.X
                    seta = etaFC1.X
                    FC2.setObjective((Bre - Areu @ uFC2 - Arexc @ sxc) @ smu + (Bri - Arixc @ sxc) @ seta, GRB.MAXIMIZE)
                    FC2.optimize()
                    su = uFC2.X

                print('test_u before FC')
                if not case.test_u(su, mpc, b_print=True, b_ellipsoid=True):
                    su = case.revise_u(su, mpc, EPS, b_print=True, b_ellipsoid=True)
                # FC: Bilinear program
                FC = gp.Model('Feasibility')
                uFC = FC.addMVar((Areu.shape[1],), lb=-float('inf'), vtype=GRB.CONTINUOUS)
                uFC.Start = su
                u_ldFC = FC.addMVar((mpc['n_t'] * mpc['n_u'],), lb=-float('inf'), vtype=GRB.CONTINUOUS)
                muFC = FC.addMVar((Bre.shape[0],), lb=-float('inf'), vtype=GRB.CONTINUOUS)
                muFC.Start = smu
                etaFC = FC.addMVar((Bri.shape[0],), lb=-float('inf'), vtype=GRB.CONTINUOUS)
                etaFC.Start = seta
                FC.addConstr(u_ldFC == mpc['u_l_predict'] - uFC - mpc['error_mu'], name='u_e')
                FC.addConstr(uFC >= mpc['u_ll'], name='u_lb')
                FC.addConstr(mpc['u_lu'] >= uFC, name='u_ub')
                FC.addConstr(mpc['u_l_predict'] - uFC >= mpc['error_lb'], name='u_e_lb')
                FC.addConstr(mpc['error_ub'] >= mpc['u_l_predict'] - uFC, name='u_e_ub')
                FC.addMQConstr(Q=mpc['error_sigma_inv'], c=None, sense='<', rhs=mpc['error_rho'], xQ_L=u_ldFC, xQ_R=u_ldFC, name='u_q')
                FC.addConstr(Arey.T @ muFC + Ariy.T @ etaFC == 0, name='de')
                FC.addConstr(muFC <= 1, name='dru')
                FC.addConstr(muFC >= -1, name='drl')
                FC.addConstr(etaFC >= 0, name='di')
                FC.setObjective((Bre - Areu @ uFC - Arexc @ sxc) @ muFC + (Bri - Arixc @ sxc) @ etaFC, GRB.MAXIMIZE)
                FC.setParam('OutputFlag', 0)
                FC.Params.TimeLimit = TimeLimitFC

                print('******************************FC******************************')
                FC.optimize()

                print('test_u after FC')
                if case.test_u(uFC.X, mpc, b_print=True, b_ellipsoid=True):
                    print('FC: su is in the uncertainty set.')
                    su = uFC.X
                    FCVal = FC.ObjVal
                else:
                    print('FC: su is not in the uncertainty set.')
                    su = case.revise_u(uFC.X, mpc, EPS, b_print=True, b_ellipsoid=True) # Revise and mountain climbing
                    FC1.setObjective((Bre - Areu @ su - Arexc @ sxc) @ muFC1 + (Bri - Arixc @ sxc) @ etaFC1, GRB.MAXIMIZE)
                    FC1.optimize()
                    smu = muFC1.X
                    seta = etaFC1.X
                    FC2.setObjective((Bre - Areu @ uFC2 - Arexc @ sxc) @ smu + (Bri - Arixc @ sxc) @ seta, GRB.MAXIMIZE)
                    FC2.optimize()
                    su = uFC2.X
                    FCVal = FC2.ObjVal

                print('FC gap (before revision): {}'.format(FC.MIPGap))
                print('FCObj (revised to be feasible): {}'.format(FCVal))
                
                if FCVal > EPS:
                    print('test_u after FC gap')
                    if not case.test_u(su, mpc, b_print=True, b_ellipsoid=True):
                        su = case.revise_u(su, mpc, EPS, b_print=True, b_ellipsoid=True)
                    ulist.append(su)
                else:
                    # SP: Mountain climbing for initiation
                    su = mpc['u_l_predict'] - mpc['error_mu'] # Initialize u
                    for i in range(3):  
                        SP1.setObjective((Bre - Areu @ su - Arexc @ sxc) @ muSP1 + (Bri - Arixc @ sxc) @ etaSP1, GRB.MAXIMIZE)
                        SP1.optimize()
                        smu = muSP1.X
                        seta = etaSP1.X
                        SP2.setObjective((Bre - Areu @ uSP2 - Arexc @ sxc) @ smu + (Bri - Arixc @ sxc) @ seta, GRB.MAXIMIZE)
                        SP2.optimize()
                        su = uSP2.X

                    print('test_u before SP')
                    if not case.test_u(su, mpc, b_print=True, b_ellipsoid=True):
                        su = case.revise_u(su, mpc, EPS, b_print=True, b_ellipsoid=True)
                    ## Sub problem: MILP
                    SP = gp.Model('Sub')
                    uSP = SP.addMVar((Areu.shape[1],), lb=-float('inf'), vtype=GRB.CONTINUOUS) 
                    # uSP.Start = su # su cannot provide a start
                    u_ldSP = SP.addMVar((Areu.shape[1],), lb=-float('inf'), vtype=GRB.CONTINUOUS)
                    ySP = SP.addMVar((Cry.shape[0],), lb=-float('inf'), vtype=GRB.CONTINUOUS)
                    muSP = SP.addMVar((Bre.shape[0],), lb=-float('inf'), vtype=GRB.CONTINUOUS) # Dual variable of equation
                    muSP.Start = smu
                    etaSP = SP.addMVar((Bri.shape[0],), lb=-float('inf'), vtype=GRB.CONTINUOUS) # Dual variable of inequality
                    etaSP.Start = seta
                    zSP = SP.addMVar((Bri.shape[0],), vtype=GRB.BINARY) # Binary variable for the big-M method
                    SP.addConstr(u_ldSP == mpc['u_l_predict'] - uSP - mpc['error_mu'], name='u_e')
                    SP.addConstr(uSP >= mpc['u_ll'], name='u_lb')
                    SP.addConstr(mpc['u_lu'] >= uSP, name='u_ub')
                    SP.addConstr(mpc['u_l_predict'] - uSP >= mpc['error_lb'], name='u_e_lb')
                    SP.addConstr(mpc['error_ub'] >= mpc['u_l_predict'] - uSP, name='u_e_ub')
                    SP.addMQConstr(Q=mpc['error_sigma_inv'], c=None, sense='<', rhs=mpc['error_rho'], xQ_L=u_ldSP, xQ_R=u_ldSP, name='u_q')
                    SP.addConstr(Areu @ uSP + Arexc @ sxc + Arey @ ySP == Bre, name='re')
                    SP.addConstr(Arixc @ sxc + Ariy @ ySP >= Bri, name='ri')
                    SP.addConstr(Arey.T @ muSP + Ariy.T @ etaSP == Cry, name='de')
                    SP.addConstr(etaSP >= 0, name='di')
                    SP.addConstr(Arixc @ sxc + Ariy @ ySP - Bri <= LargeNumber * zSP, name='Mp')
                    SP.addConstr(etaSP <= LargeNumber * (1 - zSP), name='Md')
                    SP.setObjective(Cry @ ySP, GRB.MAXIMIZE)
                    if b_display_SP:
                        SP.setParam('OutputFlag', 1)
                    else:
                        SP.setParam('OutputFlag', 0)
                    SP.Params.TimeLimit = TimeLimitSP
                    
                    print('******************************SP******************************')
                    SP.optimize()

                    print('SP gap: {}'.format(SP.MIPGap))
                    print('test_u after SP')
                    if case.test_u(uSP.X, mpc, b_print=True, b_ellipsoid=True):
                        SPVal = SP.ObjVal
                    else:
                        su = case.revise_u(uSP.X, mpc, EPS, b_print=True, b_ellipsoid=True)
                        SP1.setObjective((Bre - Areu @ su - Arexc @ sxc) @ muSP1 + (Bri - Arixc @ sxc) @ etaSP1, GRB.MAXIMIZE)
                        SP1.optimize()
                        smu = muSP1.X
                        seta = etaSP1.X
                        SP2.setObjective((Bre - Areu @ uSP2 - Arexc @ sxc) @ smu + (Bri - Arixc @ sxc) @ seta, GRB.MAXIMIZE)
                        SP2.optimize()
                        su = uSP2.X
                        SPVal = SP2.ObjVal
                    UBU[Iter] = MPObjX + SPVal * (1 + SP.MIPGap)
                    UBL[Iter] = MPObjX + SPVal

                    # Check convergence
                    print('UBU: {}, UBL: {}, LB: {}'.format(UBU[Iter], UBL[Iter], LB[Iter]))
                    if (np.min(UBU) < float('inf')) & (np.min(UBU) - LB[Iter] < Tolerance * np.min(UBU)):
                        print('The algorithm converges.')
                        break
                    else:
                        print('test_u before appending su')
                        if not case.test_u(su, mpc, b_print=True, b_ellipsoid=True):
                            su = case.revise_u(su, mpc, EPS, b_print=True, b_ellipsoid=True)
                        ulist.append(su)       
                    
                # Add constraints to MP
                MP.addConstr(zetaMP >= Cry @ yMP[Iter, :])
                MP.addConstr(Areu @ ulist[-1] + Arexc @ xcMP + Arey @ yMP[Iter, :] == Bre)
                MP.addConstr(Arixc @ xcMP + Ariy @ yMP[Iter, :] >= Bri)
                
                Iter += 1
            
            # Output
            LBUB = np.concatenate((LB[:(Iter + 1)].reshape((-1, 1)), UBU[:(Iter + 1)].reshape((-1, 1)), UBL[:(Iter + 1)].reshape((-1, 1))), axis=1)

        except gp.GurobiError as e:
            print(f"Error code {e.errno}: {e}")

        except AttributeError:
            print("Encountered an attribute error")

        time_elapsed = time.time() - time_start

        return ulist, sxb, sxc, LBUB, time_elapsed

    @staticmethod
    def c044_reconstruction(epsilon, delta, coefficients, u_data_train_n2, sxb, sxc, LBUB):
        """
        Uncertainty set reconstruction
        """

        print('(((((((((((((((((((((((((((((c044)))))))))))))))))))))))))))))')

        ## Load compact form
        Areu = coefficients['Areu']
        Arexc = coefficients['Arexc']
        Arey = coefficients['Arey']
        Arixc = coefficients['Arixc']
        Ariy = coefficients['Ariy']
        Bre = coefficients['Bre']
        Bri = coefficients['Bri']
        Cdxb = coefficients['Cdxb']
        Cdxc = coefficients['Cdxc']
        Cry = coefficients['Cry']

        ## Load u_data_train_n2
        u_data = u_data_train_n2
        num_data = u_data.shape[0]

        ## Load CCG solution
        sobj = LBUB[-1, 0]

        try:
            ## Test feasibility of points
            fmodel = gp.Model('Feasibility')
            fs = fmodel.addMVar((num_data,), lb=-float('inf'), vtype=GRB.CONTINUOUS) # Slack
            fu = fmodel.addMVar((Areu.shape[1], num_data), lb=-float('inf'), vtype=GRB.CONTINUOUS)
            fy = fmodel.addMVar((Cry.shape[0], num_data), lb=-float('inf'), vtype=GRB.CONTINUOUS)
            fmodel.addConstrs((u_data[i, :].T - fu[:, i] <= fs[i] for i in range(num_data)), name='s1')
            fmodel.addConstrs((u_data[i, :].T - fu[:, i] >= - fs[i] for i in range(num_data)), name='s2')
            fmodel.addConstrs((Areu @ fu[:, i] + Arexc @ sxc + Arey @ fy[:, i] == Bre for i in range(num_data)), name='re')
            fmodel.addConstrs((Ariy @ fy[:, i] >= Bri - Arixc @ sxc for i in range(num_data)), name='ri')
            fmodel.addConstrs((sobj >= Cdxb @ sxb + Cdxc @ sxc + Cry @ fy[:, i] for i in range(num_data)), name='obj')
            fmodel.setObjective(gp.quicksum(fs), GRB.MINIMIZE)
            fmodel.setParam('OutputFlag', 0) 
            fmodel.optimize()

            # Check and preserve the feasible points
            sfs = fs.X
            print('There are {} data in n2, where {} of them are not feasible.'.format(num_data, np.count_nonzero(sfs)))
            u_data = u_data[sfs == 0]
            num_data = u_data.shape[0]

            ## Calculate objective for all points
            omodel = gp.Model('Objective')
            os = omodel.addMVar((num_data,), lb=-float('inf'), vtype=GRB.CONTINUOUS) # Slack
            oy = omodel.addMVar((Cry.shape[0], num_data), lb=-float('inf'), vtype=GRB.CONTINUOUS)
            omodel.addConstrs((Areu @ u_data[i, :].T + Arexc @ sxc + Arey @ oy[:, i] == Bre for i in range(num_data)), name='re')
            omodel.addConstrs((Ariy @ oy[:, i] >= Bri - Arixc @ sxc for i in range(num_data)), name='ri')
            omodel.addConstrs((sobj + os[i] >= Cdxb @ sxb + Cdxc @ sxc + Cry @ oy[:, i] for i in range(num_data)), name='obj')
            omodel.setObjective(gp.quicksum(os), GRB.MINIMIZE)
            omodel.setParam('OutputFlag', 0) 
            omodel.optimize()

            # Get radius and feasible uncertainty data
            sos = os.X
            rank = CombHandler().get_rank(num_data, epsilon, delta)
            radius = sos[np.argsort(sos)[rank - 1]]
            u_data = u_data[np.argsort(sos)[:rank]]

            ## Form the model of the uncertainty set
            umodel = gp.Model('Uncertainty')
            uu = umodel.addMVar((Areu.shape[1],), lb=-float('inf'), vtype=GRB.CONTINUOUS)
            umodel.update()
            numuu = umodel.NumVars
            uy = umodel.addMVar((Cry.shape[0],), lb=-float('inf'), vtype=GRB.CONTINUOUS)
            umodel.addConstr(Areu @ uu + Arexc @ sxc + Arey @ uy == Bre, name='re')
            umodel.update()
            numue = umodel.NumConstrs
            umodel.addConstr(Ariy @ uy >= Bri - Arixc @ sxc, name='ri')
            umodel.addConstr(sobj + radius >= Cdxb @ sxb + Cdxc @ sxc + Cry @ uy, name='obj')
            umodel.setObjective(0, GRB.MINIMIZE)
            umodel.optimize()

            A = umodel.getA()
            B = np.array(umodel.getAttr('RHS', umodel.getConstrs()))
            sense = umodel.getAttr('Sense', umodel.getConstrs())
            for i, x in enumerate(sense):
                if x == '<':
                    A[i, :] = - A[i, :]
                    B[i] = - B[i]

            Aueu = A[:numue, :numuu]
            Auey = A[:numue, numuu:]
            Bue = B[:numue]
            Auiy = A[numue:, numuu:]
            Bui = B[numue:]

            coefficients['Aueu'] = Aueu
            coefficients['Auey'] = Auey
            coefficients['Auiy'] = Auiy
            coefficients['Bue'] = Bue
            coefficients['Bui'] = Bui

        except gp.GurobiError as e:
            print(f"Error code {e.errno}: {e}")

        except AttributeError:
            print("Encountered an attribute error")

        return radius, u_data, coefficients
    
    @staticmethod
    def c045_CCG_n2(LargeNumber, Tolerance, TimeLimitFC, TimeLimitSP, MaxIter, EPS, mpc, coefficients, u_data):
        """
        CCG with the reconstructed uncertainty set
        """
        print('(((((((((((((((((((((((((((((c045)))))))))))))))))))))))))))))')

        ## Time
        time_start = time.time()

        ## Load case
        case = Case()

        ## Load compact form
        Adexb = coefficients['Adexb']
        Adexc = coefficients['Adexc']
        Adixb = coefficients['Adixb']
        Adixc = coefficients['Adixc']
        Areu = coefficients['Areu']
        Arexc = coefficients['Arexc']
        Arey = coefficients['Arey']
        Arixc = coefficients['Arixc']
        Ariy = coefficients['Ariy']
        Aueu = coefficients['Aueu']
        Auey = coefficients['Auey']
        Auiy = coefficients['Auiy']
        Bde = coefficients['Bde']
        Bdi = coefficients['Bdi']
        Bre = coefficients['Bre']
        Bri = coefficients['Bri']
        Bue = coefficients['Bue']
        Bui = coefficients['Bui']   
        Cdxb = coefficients['Cdxb']
        Cdxc = coefficients['Cdxc']
        Cry = coefficients['Cry']

        ## Load uncertainty data
        num_data = u_data.shape[0]
        # Check whether they are in the uncertainty set
        tmodel = gp.Model('Test')
        ty = tmodel.addMVar((Auey.shape[1], num_data), lb=-float('inf'), vtype=GRB.CONTINUOUS)
        tmodel.addConstrs((Aueu @ u_data[i, :].T + Auey @ ty[:, i] == Bue for i in range(num_data)), name='ue')
        tmodel.addConstrs((Auiy @ ty[:, i] >= Bui for i in range(num_data)), name='ui')
        tmodel.setObjective(0, GRB.MINIMIZE)
        tmodel.setParam('OutputFlag', 0) 
        tmodel.optimize()
        if tmodel.Status == 2:
            print('All uncertainty data are feasible.')
        else:
            print('Some uncertainty data are infeasible.')

        try:
            ## Initiation
            # MP: Master problem
            MP = gp.Model('Master')
            xbMP = MP.addMVar((Cdxb.shape[0],), vtype=GRB.BINARY)
            xcMP = MP.addMVar((Cdxc.shape[0],), lb=-float('inf'), vtype=GRB.CONTINUOUS)
            yMP = MP.addMVar((MaxIter, Cry.shape[0]), lb=-float('inf'), vtype=GRB.CONTINUOUS)
            yMPdata = MP.addMVar((num_data, Cry.shape[0]), lb=-float('inf'), vtype=GRB.CONTINUOUS)
            zetaMP = MP.addVar(lb=-LargeNumber, vtype=GRB.CONTINUOUS)
            MP.setObjective(Cdxb @ xbMP + Cdxc @ xcMP + zetaMP, GRB.MINIMIZE)
            MP.addConstr(Adexb @ xbMP + Adexc @ xcMP == Bde, name='de')
            MP.addConstr(Adixb @ xbMP + Adixc @ xcMP >= Bdi, name='di')
            for i in range(num_data): # Add uncertainty data as initiation
                MP.addConstr(zetaMP >= Cry @ yMPdata[i, :], name='rcd')
                MP.addConstr(Areu @ u_data[i, :] + Arexc @ xcMP + Arey @ yMPdata[i, :] == Bre, name='red')
                MP.addConstr(Arixc @ xcMP + Ariy @ yMPdata[i, :] >= Bri, name='rid')
            MP.setParam('OutputFlag', 0) 

            # Feasibility check problem by mountain climbing
            # FC1: Fix uncertainty and optimize dual variables
            FC1 = gp.Model('Feasibility-Dual')
            muFC1 = FC1.addMVar((Bre.shape[0],), lb=-float('inf'), vtype=GRB.CONTINUOUS) # Dual variable of equation
            etaFC1 = FC1.addMVar((Bri.shape[0],), lb=-float('inf'), vtype=GRB.CONTINUOUS) # Dual variable of inequality
            FC1.addConstr(Arey.T @ muFC1 + Ariy.T @ etaFC1 == 0, name='de')
            FC1.addConstr(muFC1 <= 1, name='dru')
            FC1.addConstr(muFC1 >= -1, name='drl')
            FC1.addConstr(etaFC1 >= 0, name='di')
            FC1.setParam('OutputFlag', 0) 
            # FC2: Fix dual variables and optimize uncertainty
            FC2 = gp.Model('Feasibility-Uncertainty')
            uFC2 = FC2.addMVar((Areu.shape[1],), lb=-float('inf'), vtype=GRB.CONTINUOUS)
            yFC2 = FC2.addMVar((Auey.shape[1],), lb=-float('inf'), vtype=GRB.CONTINUOUS) # Second-stage variable for defining the uncertainty set
            FC2.addConstr(uFC2 >= mpc['u_ll'], name='u_lb')
            FC2.addConstr(mpc['u_lu'] >= uFC2, name='u_ub')
            FC2.addConstr(mpc['u_l_predict'] - uFC2 >= mpc['error_lb'], name='u_e_lb')
            FC2.addConstr(mpc['error_ub'] >= mpc['u_l_predict'] - uFC2, name='u_e_ub')
            FC2.addConstr(Aueu @ uFC2 + Auey @ yFC2 == Bue, 'ue')
            FC2.addConstr(Auiy @ yFC2 >= Bui, 'ui')
            FC2.setParam('OutputFlag', 0) 

            ## Subproblem by mountain climbing
            # SP1: Fix uncertainty and optimize dual variables
            SP1 = gp.Model('Subproblem-Dual')
            muSP1 = SP1.addMVar((Bre.shape[0],), lb=-float('inf'), vtype=GRB.CONTINUOUS) # Dual variable of equation
            etaSP1 = SP1.addMVar((Bri.shape[0],), lb=-float('inf'), vtype=GRB.CONTINUOUS) # Dual variable of inequality
            SP1.addConstr(Arey.T @ muSP1 + Ariy.T @ etaSP1 == Cry, name='de')
            SP1.addConstr(etaSP1 >= 0, name='di')
            SP1.setParam('OutputFlag', 0) 
            # SP2: Fix dual variables and optimize uncertainty
            SP2 = gp.Model('Subproblem-Uncertainty')
            uSP2 = SP2.addMVar((Areu.shape[1],), lb=-float('inf'), vtype=GRB.CONTINUOUS)
            ySP2 = SP2.addMVar((Auey.shape[1],), lb=-float('inf'), vtype=GRB.CONTINUOUS) # Second-stage variable for defining the uncertainty set
            SP2.addConstr(uSP2 >= mpc['u_ll'], name='u_lb')
            SP2.addConstr(mpc['u_lu'] >= uSP2, name='u_ub')
            SP2.addConstr(mpc['u_l_predict'] - uSP2 >= mpc['error_lb'], name='u_e_lb')
            SP2.addConstr(mpc['error_ub'] >= mpc['u_l_predict'] - uSP2, name='u_e_ub')
            SP2.addConstr(Aueu @ uSP2 + Auey @ ySP2 == Bue, 'ue')
            SP2.addConstr(Auiy @ ySP2 >= Bui, 'ui')
            SP2.setParam('OutputFlag', 0) 

            # Other initiation
            LB = -float('inf') * np.ones((MaxIter,))
            UBU = float('inf') * np.ones((MaxIter,)) # Theoretical upper bound
            UBL = float('inf') * np.ones((MaxIter,)) # Founded upper bound
            ulist = [] # The list of worst-case uncertainty
            Iter = 0

            ## Iteration
            while True:
                print('**************************************************************')
                print('Begin iteration: {}'.format(Iter))

                print('******************************MP******************************')
                MP.optimize()
                sxb = xbMP.X
                sxc = xcMP.X
                LB[Iter] = MP.ObjVal
                MPObjX = Cdxb @ sxb + Cdxc @ sxc
                if np.max(yMPdata.X @ Cry) > np.max(yMP.X @ Cry):
                    sy = yMPdata.X[np.argmax(yMPdata.X @ Cry), :]
                else:
                    sy = yMP.X[np.argmax(yMP.X @ Cry), :]

                print('LB: {} >= first-stage cost: {}'.format(LB[Iter], MPObjX))

                if (Iter > 0) & (LB[Iter] < LB[Iter - 1] * (1 + EPS)):
                    print('Cannot improve. Terminate.')
                    break

                # FC: Mountain climbing as initiation
                su = mpc['u_l_predict'] - mpc['error_mu'] # Initialize u
                for i in range(3):
                    FC1.setObjective((Bre - Areu @ su - Arexc @ sxc) @ muFC1 + (Bri - Arixc @ sxc) @ etaFC1, GRB.MAXIMIZE)
                    FC1.optimize()
                    smu = muFC1.X
                    seta = etaFC1.X
                    FC2.setObjective((Bre - Areu @ uFC2 - Arexc @ sxc) @ smu + (Bri - Arixc @ sxc) @ seta, GRB.MAXIMIZE)
                    FC2.optimize()
                    su = uFC2.X

                print('test_u before FC')
                if not case.test_u(su, mpc, b_print=True, b_ellipsoid=False):
                    su = case.revise_u(su, mpc, EPS, b_print=True, b_ellipsoid=False)
                # FC: Bilinear program
                FC = gp.Model('Feasibility')
                uFC = FC.addMVar((Areu.shape[1],), lb=-float('inf'), vtype=GRB.CONTINUOUS)
                uFC.Start = su
                yFC = FC.addMVar((Auey.shape[1],), lb=-float('inf'), vtype=GRB.CONTINUOUS) # Second-stage variable for defining the uncertainty set
                muFC = FC.addMVar((Bre.shape[0],), lb=-float('inf'), vtype=GRB.CONTINUOUS)
                muFC.Start = smu
                etaFC = FC.addMVar((Bri.shape[0],), lb=-float('inf'), vtype=GRB.CONTINUOUS)
                etaFC.Start = seta
                FC.addConstr(uFC >= mpc['u_ll'], name='u_lb')
                FC.addConstr(mpc['u_lu'] >= uFC, name='u_ub')
                FC.addConstr(mpc['u_l_predict'] - uFC >= mpc['error_lb'], name='u_e_lb')
                FC.addConstr(mpc['error_ub'] >= mpc['u_l_predict'] - uFC, name='u_e_ub')
                FC.addConstr(Aueu @ uFC + Auey @ yFC == Bue, 'ue')
                FC.addConstr(Auiy @ yFC >= Bui, 'ui')
                FC.addConstr(Arey.T @ muFC + Ariy.T @ etaFC == 0, name='de')
                FC.addConstr(muFC <= 1, name='dru')
                FC.addConstr(muFC >= -1, name='drl')
                FC.addConstr(etaFC >= 0, name='di')
                FC.setObjective((Bre - Areu @ uFC - Arexc @ sxc) @ muFC + (Bri - Arixc @ sxc) @ etaFC, GRB.MAXIMIZE)
                FC.setParam('OutputFlag', 0)
                FC.Params.TimeLimit = TimeLimitFC

                print('******************************FC******************************')
                FC.optimize()

                print('test_u after FC')
                if case.test_u(uFC.X, mpc, b_print=True, b_ellipsoid=False):
                    print('FC: su is in the uncertainty set.')
                    su = uFC.X
                    FCVal = FC.ObjVal
                else:
                    print('FC: su is not in the uncertainty set.')
                    su = case.revise_u(uFC.X, mpc, EPS, b_print=True, b_ellipsoid=False) # Revise and mountain climbing
                    FC1.setObjective((Bre - Areu @ su - Arexc @ sxc) @ muFC1 + (Bri - Arixc @ sxc) @ etaFC1, GRB.MAXIMIZE)
                    FC1.optimize()
                    smu = muFC1.X
                    seta = etaFC1.X
                    FC2.setObjective((Bre - Areu @ uFC2 - Arexc @ sxc) @ smu + (Bri - Arixc @ sxc) @ seta, GRB.MAXIMIZE)
                    FC2.optimize()
                    su = uFC2.X
                    FCVal = FC2.ObjVal

                print('FC gap (before revision): {}'.format(FC.MIPGap))
                print('FCObj (revised to be feasible): {}'.format(FCVal))
                
                if FCVal > EPS:
                    print('test_u after FC gap')
                    if not case.test_u(su, mpc, b_print=True, b_ellipsoid=False):
                        su = case.revise_u(su, mpc, EPS, b_print=True, b_ellipsoid=False)
                    ulist.append(su)
                else:
                    # SP: Mountain climbing for initiation
                    su = mpc['u_l_predict'] - mpc['error_mu'] # Initialize u
                    for i in range(3):  
                        SP1.setObjective((Bre - Areu @ su - Arexc @ sxc) @ muSP1 + (Bri - Arixc @ sxc) @ etaSP1, GRB.MAXIMIZE)
                        SP1.optimize()
                        smu = muSP1.X
                        seta = etaSP1.X
                        SP2.setObjective((Bre - Areu @ uSP2 - Arexc @ sxc) @ smu + (Bri - Arixc @ sxc) @ seta, GRB.MAXIMIZE)
                        SP2.optimize()
                        su = uSP2.X

                    print('test_u before SP')
                    if not case.test_u(su, mpc, b_print=True, b_ellipsoid=False):
                        su = case.revise_u(su, mpc, EPS, b_print=True, b_ellipsoid=False)
                    ## Sub problem: MILP
                    SP = gp.Model('Sub')
                    uSP = SP.addMVar((Areu.shape[1],), lb=-float('inf'), vtype=GRB.CONTINUOUS) 
                    # uSP.Start = su # su cannot provide a start
                    ySPu = SP.addMVar((Auey.shape[1],), lb=-float('inf'), vtype=GRB.CONTINUOUS) # Second-stage variable for defining the uncertainty set
                    ySP = SP.addMVar((Cry.shape[0],), lb=-float('inf'), vtype=GRB.CONTINUOUS)
                    muSP = SP.addMVar((Bre.shape[0],), lb=-float('inf'), vtype=GRB.CONTINUOUS) # Dual variable of equation
                    muSP.Start = smu
                    etaSP = SP.addMVar((Bri.shape[0],), lb=-float('inf'), vtype=GRB.CONTINUOUS) # Dual variable of inequality
                    etaSP.Start = seta
                    zSP = SP.addMVar((Bri.shape[0],), vtype=GRB.BINARY) # Binary variable for the big-M method
                    SP.addConstr(uSP >= mpc['u_ll'], name='u_lb')
                    SP.addConstr(mpc['u_lu'] >= uSP, name='u_ub')
                    SP.addConstr(mpc['u_l_predict'] - uSP >= mpc['error_lb'], name='u_e_lb')
                    SP.addConstr(mpc['error_ub'] >= mpc['u_l_predict'] - uSP, name='u_e_ub')
                    SP.addConstr(Aueu @ uSP + Auey @ ySPu == Bue, 'ue')
                    SP.addConstr(Auiy @ ySPu >= Bui, 'ui')
                    SP.addConstr(Areu @ uSP + Arexc @ sxc + Arey @ ySP == Bre, name='re')
                    SP.addConstr(Arixc @ sxc + Ariy @ ySP >= Bri, name='ri')
                    SP.addConstr(Arey.T @ muSP + Ariy.T @ etaSP == Cry, name='de')
                    SP.addConstr(etaSP >= 0, name='di')
                    SP.addConstr(Arixc @ sxc + Ariy @ ySP - Bri <= LargeNumber * zSP, name='Mp')
                    SP.addConstr(etaSP <= LargeNumber * (1 - zSP), name='Md')
                    SP.setObjective(Cry @ ySP, GRB.MAXIMIZE)
                    SP.setParam('OutputFlag', 1)
                    SP.Params.TimeLimit = TimeLimitSP
                    
                    print('******************************SP******************************')
                    SP.optimize()

                    print('SP gap: {}'.format(SP.MIPGap))
                    print('test_u after SP')
                    if case.test_u(uSP.X, mpc, b_print=True, b_ellipsoid=False):
                        SPVal = SP.ObjVal
                    else:
                        su = case.revise_u(uSP.X, mpc, EPS, b_print=True, b_ellipsoid=False)
                        SP1.setObjective((Bre - Areu @ su - Arexc @ sxc) @ muSP1 + (Bri - Arixc @ sxc) @ etaSP1, GRB.MAXIMIZE)
                        SP1.optimize()
                        smu = muSP1.X
                        seta = etaSP1.X
                        SP2.setObjective((Bre - Areu @ uSP2 - Arexc @ sxc) @ smu + (Bri - Arixc @ sxc) @ seta, GRB.MAXIMIZE)
                        SP2.optimize()
                        su = uSP2.X
                        SPVal = SP2.ObjVal
                    UBU[Iter] = MPObjX + SPVal * (1 + SP.MIPGap)
                    UBL[Iter] = MPObjX + SPVal

                    # Check convergence
                    print('UBU: {}, UBL: {}, LB: {}'.format(UBU[Iter], UBL[Iter], LB[Iter]))
                    if (np.min(UBU) < float('inf')) & (np.min(UBU) - LB[Iter] < Tolerance * np.min(UBU)):
                        print('The algorithm converges.')
                        break
                    else:
                        print('test_u before appending su')
                        if not case.test_u(su, mpc, b_print=True, b_ellipsoid=False):
                            su = case.revise_u(su, mpc, EPS, b_print=True, b_ellipsoid=False)
                        ulist.append(su)       
                    
                # Add constraints to MP
                MP.addConstr(zetaMP >= Cry @ yMP[Iter, :])
                MP.addConstr(Areu @ ulist[-1] + Arexc @ xcMP + Arey @ yMP[Iter, :] == Bre)
                MP.addConstr(Arixc @ xcMP + Ariy @ yMP[Iter, :] >= Bri)
                
                Iter += 1
            
            # Output
            LBUB = np.concatenate((LB[:(Iter + 1)].reshape((-1, 1)), UBU[:(Iter + 1)].reshape((-1, 1)), UBL[:(Iter + 1)].reshape((-1, 1))), axis=1)

        except gp.GurobiError as e:
            print(f"Error code {e.errno}: {e}")

        except AttributeError:
            print("Encountered an attribute error")

        time_elapsed = time.time() - time_start

        interpret = {}
        interpret['x_og'] = sxb[:(mpc['n_t'] * mpc['n_g'])].reshape((mpc['n_t'], mpc['n_g']))
        interpret['x_pg'] = sxc[:(mpc['n_t'] * mpc['n_g'])].reshape((mpc['n_t'], mpc['n_g']))
        interpret['x_rp'] = sxc[(mpc['n_t'] * mpc['n_g']):(mpc['n_t'] * mpc['n_g'] * 2)].reshape((mpc['n_t'], mpc['n_g']))
        interpret['x_rn'] = sxc[(mpc['n_t'] * mpc['n_g'] * 2):].reshape((mpc['n_t'], mpc['n_g']))
        interpret['y_rp'] = sy[:(mpc['n_t'] * mpc['n_g'])].reshape((mpc['n_t'], mpc['n_g']))
        interpret['y_rn'] = sy[(mpc['n_t'] * mpc['n_g']):].reshape((mpc['n_t'], mpc['n_g']))

        return u_data, ulist, sxb, sxc, LBUB, time_elapsed, interpret
    
    @staticmethod
    def c046_evaluate(u_data, coefficients, sxb, sxc, LBUB):
        """
        Use validation/test dataset to evaluate the obtained strategy
        """
        print('(((((((((((((((((((((((((((((c046)))))))))))))))))))))))))))))')

        ## Load compact form
        Areu = coefficients['Areu']
        Arexc = coefficients['Arexc']
        Arey = coefficients['Arey']
        Arixc = coefficients['Arixc']
        Ariy = coefficients['Ariy']
        Bre = coefficients['Bre']
        Bri = coefficients['Bri']
        Cdxb = coefficients['Cdxb']
        Cdxc = coefficients['Cdxc']
        Cry = coefficients['Cry']

        num_data = u_data.shape[0]

        ## Load CCG solution
        sobj = LBUB[-1, 0]

        ## Cost vector to indicate infeasibility/objective
        cost = np.ones((num_data,)) * float('inf')

        try:
            ## Test feasibility of points
            fmodel = gp.Model('Feasibility')
            fs = fmodel.addMVar((num_data,), lb=-float('inf'), vtype=GRB.CONTINUOUS) # Slack
            fu = fmodel.addMVar((Areu.shape[1], num_data), lb=-float('inf'), vtype=GRB.CONTINUOUS)
            fy = fmodel.addMVar((Cry.shape[0], num_data), lb=-float('inf'), vtype=GRB.CONTINUOUS)
            fmodel.addConstrs((u_data[i, :].T - fu[:, i] <= fs[i] for i in range(num_data)), name='s1')
            fmodel.addConstrs((u_data[i, :].T - fu[:, i] >= - fs[i] for i in range(num_data)), name='s2')
            fmodel.addConstrs((Areu @ fu[:, i] + Arexc @ sxc + Arey @ fy[:, i] == Bre for i in range(num_data)), name='re')
            fmodel.addConstrs((Ariy @ fy[:, i] >= Bri - Arixc @ sxc for i in range(num_data)), name='ri')
            fmodel.addConstrs((sobj >= Cdxb @ sxb + Cdxc @ sxc + Cry @ fy[:, i] for i in range(num_data)), name='obj')
            fmodel.setObjective(gp.quicksum(fs), GRB.MINIMIZE)
            fmodel.setParam('OutputFlag', 0) 
            fmodel.optimize()

            # Check and preserve the feasible points
            sfs = fs.X
            print('There are {} out of {} data that are not feasible.'.format(np.count_nonzero(sfs), num_data))
            u_data = u_data[sfs == 0]
            num_data = u_data.shape[0]

            ## Calculate objective for all points
            omodel = gp.Model('Objective')
            os = omodel.addMVar((num_data,), lb=-float('inf'), vtype=GRB.CONTINUOUS) # Slack
            oy = omodel.addMVar((Cry.shape[0], num_data), lb=-float('inf'), vtype=GRB.CONTINUOUS)
            omodel.addConstrs((Areu @ u_data[i, :].T + Arexc @ sxc + Arey @ oy[:, i] == Bre for i in range(num_data)), name='re')
            omodel.addConstrs((Ariy @ oy[:, i] >= Bri - Arixc @ sxc for i in range(num_data)), name='ri')
            omodel.addConstrs((sobj + os[i] >= Cdxb @ sxb + Cdxc @ sxc + Cry @ oy[:, i] for i in range(num_data)), name='obj')
            omodel.setObjective(gp.quicksum(os), GRB.MINIMIZE)
            omodel.setParam('OutputFlag', 0) 
            omodel.optimize()

            # Get radius and feasible uncertainty data
            cost[:num_data] = sobj + os.X
            cost = np.sort(cost)[::-1]

        except gp.GurobiError as e:
            print(f"Error code {e.errno}: {e}")

        except AttributeError:
            print("Encountered an attribute error")

        return cost
    
    @staticmethod
    def c046_evaluate_order(u_data, coefficients, sxb, sxc, LBUB):
        """
        Use validation/test dataset to evaluate the obtained strategy
        """
        print('(((((((((((((((((((((((((((((c046)))))))))))))))))))))))))))))')

        ## Load compact form
        Areu = coefficients['Areu']
        Arexc = coefficients['Arexc']
        Arey = coefficients['Arey']
        Arixc = coefficients['Arixc']
        Ariy = coefficients['Ariy']
        Bre = coefficients['Bre']
        Bri = coefficients['Bri']
        Cdxb = coefficients['Cdxb']
        Cdxc = coefficients['Cdxc']
        Cry = coefficients['Cry']

        num_data = u_data.shape[0]

        ## Load CCG solution
        sobj = LBUB[-1, 0]

        ## Cost vector to indicate infeasibility/objective
        cost = np.zeros((num_data,))

        try:
            ## Test feasibility of points
            fmodel = gp.Model('Feasibility')
            fs = fmodel.addMVar((num_data,), lb=-float('inf'), vtype=GRB.CONTINUOUS) # Slack
            fu = fmodel.addMVar((Areu.shape[1], num_data), lb=-float('inf'), vtype=GRB.CONTINUOUS)
            fy = fmodel.addMVar((Cry.shape[0], num_data), lb=-float('inf'), vtype=GRB.CONTINUOUS)
            fmodel.addConstrs((u_data[i, :].T - fu[:, i] <= fs[i] for i in range(num_data)), name='s1')
            fmodel.addConstrs((u_data[i, :].T - fu[:, i] >= - fs[i] for i in range(num_data)), name='s2')
            fmodel.addConstrs((Areu @ fu[:, i] + Arexc @ sxc + Arey @ fy[:, i] == Bre for i in range(num_data)), name='re')
            fmodel.addConstrs((Ariy @ fy[:, i] >= Bri - Arixc @ sxc for i in range(num_data)), name='ri')
            fmodel.addConstrs((sobj >= Cdxb @ sxb + Cdxc @ sxc + Cry @ fy[:, i] for i in range(num_data)), name='obj')
            fmodel.setObjective(gp.quicksum(fs), GRB.MINIMIZE)
            fmodel.setParam('OutputFlag', 0) 
            fmodel.optimize()

            # Check and preserve the feasible points
            sfs = fs.X
            print('There are {} out of {} data that are not feasible.'.format(np.count_nonzero(sfs), num_data))
            cost[sfs > 0] = float('inf')
            num_data = u_data.shape[0]

            num_list = [i for i in range(num_data) if sfs[i] == 0]

            ## Calculate objective for all points
            omodel = gp.Model('Objective')
            os = omodel.addMVar((num_data,), lb=-1e10, vtype=GRB.CONTINUOUS) # Slack
            oy = omodel.addMVar((Cry.shape[0], num_data), lb=-float('inf'), vtype=GRB.CONTINUOUS)
            omodel.addConstrs((Areu @ u_data[i, :].T + Arexc @ sxc + Arey @ oy[:, i] == Bre for i in num_list), name='re')
            omodel.addConstrs((Ariy @ oy[:, i] >= Bri - Arixc @ sxc for i in num_list), name='ri')
            omodel.addConstrs((sobj + os[i] >= Cdxb @ sxb + Cdxc @ sxc + Cry @ oy[:, i] for i in num_list), name='obj')
            omodel.setObjective(gp.quicksum(os), GRB.MINIMIZE)
            omodel.setParam('OutputFlag', 0) 
            omodel.optimize()

            # Get radius and feasible uncertainty data
            cost[sfs == 0] = sobj + os.X[sfs == 0]

        except gp.GurobiError as e:
            print(f"Error code {e.errno}: {e}")

        except AttributeError:
            print("Encountered an attribute error")

        return cost
    
    @staticmethod
    def weight2cost(parameter, weight, index_u_l_predict=0, name_case='case_ieee30'):
        """
        Combine c032, c041-c046
        """
        optimization = Optimization()

        b_use_n2 = parameter['b_use_n2']
        b_display_SP = parameter['b_display_SP']
        num_groups = parameter['num_groups']
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

        train_real, train_predict, train_n1_real, train_n1_predict, train_n2_real, train_n2_predict, validation_real, validation_predict, test_real, test_predict, error_bounds = optimization.c032_calculate_weight(num_groups, weight)
        error_mu, error_sigma, error_rho = optimization.c041_initial_uncertainty(b_use_n2, horizon, epsilon, delta, u_select, train_n1_real, train_n1_predict, train_n2_real, train_n2_predict)
        u_l_predict = validation_predict[(index_u_l_predict * horizon):((index_u_l_predict + 1) * horizon)]
        mpc, coefficients, u_data_train, u_data_train_n2, u_data_validation, u_data_test, u_data_train_original = optimization.c042_dispatch_model(u_select, error_mu, error_sigma, error_rho, error_bounds, EPS, train_real, train_predict, train_n2_real, train_n2_predict, validation_real, validation_predict, test_real, test_predict, u_l_predict, name_case)

        # _, sxb1, sxc1, LBUB1, time1 = optimization.c043_CCG_n1_faster(LargeNumber, Tolerance, TimeLimitFC, TimeLimitSP, MaxIter, EPS, mpc, coefficients, u_data_train, b_display_SP)
        # _, u_data, coefficients = optimization.c044_reconstruction_faster(epsilon, delta, coefficients, u_data_train_n2, sxb1, sxc1, LBUB1)
        # _, _, sxb2, sxc2, LBUB2, time2, interpret = optimization.c045_CCG_n2_faster(LargeNumber, Tolerance, TimeLimitFC, TimeLimitSP, MaxIter, EPS, mpc, coefficients, u_data)
        # validation_cost = optimization.c046_evaluate_faster(u_data_validation, coefficients, sxb2, sxc2, LBUB2)
        # test_cost = optimization.c046_evaluate_faster(u_data_test, coefficients, sxb2, sxc2, LBUB2)
        if name_case == 'case_ieee30':
            _, sxb1, sxc1, LBUB1, time1 = optimization.c043_CCG_n1(LargeNumber, Tolerance, TimeLimitFC, TimeLimitSP, MaxIter, EPS, mpc, coefficients, u_data_train, b_display_SP)
            _, u_data, coefficients = optimization.c044_reconstruction(epsilon, delta, coefficients, u_data_train_n2, sxb1, sxc1, LBUB1)
            _, _, sxb2, sxc2, LBUB2, time2, interpret = optimization.c045_CCG_n2(LargeNumber, Tolerance, TimeLimitFC, TimeLimitSP, MaxIter, EPS, mpc, coefficients, u_data)
            validation_cost = optimization.c046_evaluate(u_data_validation, coefficients, sxb2, sxc2, LBUB2)
            test_cost = optimization.c046_evaluate(u_data_test, coefficients, sxb2, sxc2, LBUB2)
            train_cost = optimization.c046_evaluate_order(u_data_train_original, coefficients, sxb2, sxc2, LBUB2)
        else:
            _, sxb1, sxc1, LBUB1, time1 = optimization.c043_CCG_n1_faster(LargeNumber, Tolerance, TimeLimitFC, TimeLimitSP, MaxIter, EPS, mpc, coefficients, u_data_train, b_display_SP)
            _, u_data, coefficients = optimization.c044_reconstruction_faster(epsilon, delta, coefficients, u_data_train_n2, sxb1, sxc1, LBUB1)
            _, _, sxb2, sxc2, LBUB2, time2, interpret = optimization.c045_CCG_n2_faster(LargeNumber, Tolerance, TimeLimitFC, TimeLimitSP, MaxIter, EPS, mpc, coefficients, u_data)
            validation_cost = optimization.c046_evaluate_faster(u_data_validation, coefficients, sxb2, sxc2, LBUB2)
            test_cost = optimization.c046_evaluate_faster(u_data_test, coefficients, sxb2, sxc2, LBUB2)
            train_cost = optimization.c046_evaluate_faster_order(u_data_train_original, coefficients, sxb2, sxc2, LBUB2)
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
        optimization = Optimization()

        b_use_n2 = parameter['b_use_n2']
        b_display_SP = parameter['b_display_SP']
        num_groups = parameter['num_groups']
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

        train_real, train_predict, train_n1_real, train_n1_predict, train_n2_real, train_n2_predict, validation_real, validation_predict, test_real, test_predict, error_bounds = optimization.c032_calculate_weight(num_groups, weight)
        error_mu, error_sigma, error_rho = optimization.c041_initial_uncertainty(b_use_n2, horizon, epsilon, delta, u_select, train_n1_real, train_n1_predict, train_n2_real, train_n2_predict)
        u_l_predict = validation_predict[(index_u_l_predict * horizon):((index_u_l_predict + 1) * horizon)]
        mpc, coefficients, u_data_train, u_data_train_n2, u_data_validation, u_data_test, u_data_train_original = optimization.c042_dispatch_model(u_select, error_mu, error_sigma, error_rho, error_bounds, EPS, train_real, train_predict, train_n2_real, train_n2_predict, validation_real, validation_predict, test_real, test_predict, u_l_predict, name_case)

        # _, sxb1, sxc1, LBUB1, time1 = optimization.c043_CCG_n1_faster(LargeNumber, Tolerance, TimeLimitFC, TimeLimitSP, MaxIter, EPS, mpc, coefficients, u_data_train, b_display_SP)
        # validation_cost = optimization.c046_evaluate_faster(u_data_validation, coefficients, sxb1, sxc1, LBUB1)
        # test_cost = optimization.c046_evaluate_faster(u_data_test, coefficients, sxb1, sxc1, LBUB1)
        if name_case == 'case_ieee30':
            _, sxb1, sxc1, LBUB1, time1 = optimization.c043_CCG_n1(LargeNumber, Tolerance, TimeLimitFC, TimeLimitSP, MaxIter, EPS, mpc, coefficients, u_data_train, b_display_SP)
            validation_cost = optimization.c046_evaluate(u_data_validation, coefficients, sxb1, sxc1, LBUB1)
            test_cost = optimization.c046_evaluate(u_data_test, coefficients, sxb1, sxc1, LBUB1)
        else:
            _, sxb1, sxc1, LBUB1, time1 = optimization.c043_CCG_n1_faster(LargeNumber, Tolerance, TimeLimitFC, TimeLimitSP, MaxIter, EPS, mpc, coefficients, u_data_train, b_display_SP)
            validation_cost = optimization.c046_evaluate_faster(u_data_validation, coefficients, sxb1, sxc1, LBUB1)
            test_cost = optimization.c046_evaluate_faster(u_data_test, coefficients, sxb1, sxc1, LBUB1)

        print('Calculated bound: {}'.format(LBUB1[-1, 0]))
        print('Validation bound:')
        print(validation_cost[:3])

        return validation_cost, test_cost, sxb1, sxc1, LBUB1, time1
    
    @staticmethod
    def weight2cost_RO(parameter, weight, index_u_l_predict=0, name_case='case_ieee30'):
        """
        Combine c032, c041-c043, c046
        """
        optimization = Optimization()

        b_use_n2 = parameter['b_use_n2']
        b_display_SP = parameter['b_display_SP']
        num_groups = parameter['num_groups']
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

        train_real, train_predict, train_n1_real, train_n1_predict, train_n2_real, train_n2_predict, validation_real, validation_predict, test_real, test_predict, error_bounds = optimization.c032_calculate_weight(num_groups, weight)
        error_mu, error_sigma, error_rho = optimization.c041_initial_uncertainty_RO(b_use_n2, horizon, epsilon, delta, u_select, train_n1_real, train_n1_predict, train_n2_real, train_n2_predict)
        u_l_predict = validation_predict[(index_u_l_predict * horizon):((index_u_l_predict + 1) * horizon)]
        mpc, coefficients, u_data_train, u_data_train_n2, u_data_validation, u_data_test, u_data_train_original = optimization.c042_dispatch_model(u_select, error_mu, error_sigma, error_rho, error_bounds, EPS, train_real, train_predict, train_n2_real, train_n2_predict, validation_real, validation_predict, test_real, test_predict, u_l_predict, name_case)

        # _, sxb1, sxc1, LBUB1, time1 = optimization.c043_CCG_n1_faster(LargeNumber, Tolerance, TimeLimitFC, TimeLimitSP, MaxIter, EPS, mpc, coefficients, u_data_train, b_display_SP)
        # validation_cost = optimization.c046_evaluate_faster(u_data_validation, coefficients, sxb1, sxc1, LBUB1)
        # test_cost = optimization.c046_evaluate_faster(u_data_test, coefficients, sxb1, sxc1, LBUB1)
        if name_case == 'case_ieee30':
            _, sxb1, sxc1, LBUB1, time1 = optimization.c043_CCG_n1(LargeNumber, Tolerance, TimeLimitFC, TimeLimitSP, MaxIter, EPS, mpc, coefficients, u_data_train, b_display_SP)
            validation_cost = optimization.c046_evaluate(u_data_validation, coefficients, sxb1, sxc1, LBUB1)
            test_cost = optimization.c046_evaluate(u_data_test, coefficients, sxb1, sxc1, LBUB1)
        else:
            _, sxb1, sxc1, LBUB1, time1 = optimization.c043_CCG_n1_faster(LargeNumber, Tolerance, TimeLimitFC, TimeLimitSP, MaxIter, EPS, mpc, coefficients, u_data_train, b_display_SP)
            validation_cost = optimization.c046_evaluate_faster(u_data_validation, coefficients, sxb1, sxc1, LBUB1)
            test_cost = optimization.c046_evaluate_faster(u_data_test, coefficients, sxb1, sxc1, LBUB1)

        print('Calculated bound: {}'.format(LBUB1[-1, 0]))
        print('Validation bound:')
        print(validation_cost[:3])

        return validation_cost, test_cost, sxb1, sxc1, LBUB1, time1
    
    @staticmethod
    def weight2cost_dataRO(num_list, parameter, weight, index_u_l_predict=0, name_case='case_ieee30'):
        """
        Combine c032, c041-c043, c046
        """
        optimization = Optimization()

        b_use_n2 = parameter['b_use_n2']
        b_display_SP = parameter['b_display_SP']
        num_groups = parameter['num_groups']
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

        train_real, train_predict, train_n1_real, train_n1_predict, train_n2_real, train_n2_predict, validation_real, validation_predict, test_real, test_predict, error_bounds = optimization.c032_calculate_weight(num_groups, weight)
        error_mu, error_sigma, error_rho = optimization.c041_initial_uncertainty(b_use_n2, horizon, epsilon, delta, u_select, train_n1_real, train_n1_predict, train_n2_real, train_n2_predict)
        u_l_predict = validation_predict[(index_u_l_predict * horizon):((index_u_l_predict + 1) * horizon)]
        mpc, coefficients, u_data_train, u_data_train_n2, u_data_validation, u_data_test, u_data_train_original = optimization.c042_dispatch_model(u_select, error_mu, error_sigma, error_rho, error_bounds, EPS, train_real, train_predict, train_n2_real, train_n2_predict, validation_real, validation_predict, test_real, test_predict, u_l_predict, name_case)

        u_data_train_original = u_data_train_original[num_list, :]

        sxb1, sxc1, LBUB1, time1 = optimization.c043_list(LargeNumber, Tolerance, TimeLimitFC, TimeLimitSP, MaxIter, EPS, mpc, coefficients, u_data_train_original, b_display_SP)

        validation_cost = optimization.c046_evaluate_faster(u_data_validation, coefficients, sxb1, sxc1, LBUB1)
        test_cost = optimization.c046_evaluate_faster(u_data_test, coefficients, sxb1, sxc1, LBUB1)

        print('Calculated bound: {}'.format(LBUB1[-1, 0]))
        print('Validation bound:')
        print(validation_cost[:3])

        return validation_cost, test_cost, sxb1, sxc1, LBUB1, time1
    
    @staticmethod
    def weight2cost_SPapprox(parameter, weight, index_u_l_predict=0, name_case='case_ieee30'):
        """
        Combine c032, c041-c043, c046
        """
        optimization = Optimization()

        b_use_n2 = parameter['b_use_n2']
        b_display_SP = parameter['b_display_SP']
        num_groups = parameter['num_groups']
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

        train_real, train_predict, train_n1_real, train_n1_predict, train_n2_real, train_n2_predict, validation_real, validation_predict, test_real, test_predict, error_bounds = optimization.c032_calculate_weight(num_groups, weight)
        error_mu, error_sigma, error_rho = optimization.c041_initial_uncertainty(b_use_n2, horizon, epsilon, delta, u_select, train_n1_real, train_n1_predict, train_n2_real, train_n2_predict)
        u_l_predict = validation_predict[(index_u_l_predict * horizon):((index_u_l_predict + 1) * horizon)]
        mpc, coefficients, u_data_train, u_data_train_n2, u_data_validation, u_data_test, u_data_train_original = optimization.c042_dispatch_model(u_select, error_mu, error_sigma, error_rho, error_bounds, EPS, train_real, train_predict, train_n2_real, train_n2_predict, validation_real, validation_predict, test_real, test_predict, u_l_predict, name_case)

        sxb1, sxc1, LBUB1, time1 = optimization.c043_list_approx(epsilon, LargeNumber, Tolerance, TimeLimitFC, TimeLimitSP, MaxIter, EPS, mpc, coefficients, u_data_train_original, b_display_SP)

        validation_cost = optimization.c046_evaluate_faster(u_data_validation, coefficients, sxb1, sxc1, LBUB1)
        test_cost = optimization.c046_evaluate_faster(u_data_test, coefficients, sxb1, sxc1, LBUB1)

        print('Calculated bound: {}'.format(LBUB1[-1, 0]))
        print('Validation bound:')
        print(validation_cost[:3])

        return validation_cost, test_cost, sxb1, sxc1, LBUB1, time1
    
    @staticmethod
    def weight2cost_SP(parameter, weight, index_u_l_predict=0, name_case='case_ieee30'):
        """
        Combine c032, c041-c043, c046
        """
        optimization = Optimization()

        b_use_n2 = parameter['b_use_n2']
        b_display_SP = parameter['b_display_SP']
        num_groups = parameter['num_groups']
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

        train_real, train_predict, train_n1_real, train_n1_predict, train_n2_real, train_n2_predict, validation_real, validation_predict, test_real, test_predict, error_bounds = optimization.c032_calculate_weight(num_groups, weight)
        error_mu, error_sigma, error_rho = optimization.c041_initial_uncertainty(b_use_n2, horizon, epsilon, delta, u_select, train_n1_real, train_n1_predict, train_n2_real, train_n2_predict)
        u_l_predict = validation_predict[(index_u_l_predict * horizon):((index_u_l_predict + 1) * horizon)]
        mpc, coefficients, u_data_train, u_data_train_n2, u_data_validation, u_data_test, u_data_train_original = optimization.c042_dispatch_model(u_select, error_mu, error_sigma, error_rho, error_bounds, EPS, train_real, train_predict, train_n2_real, train_n2_predict, validation_real, validation_predict, test_real, test_predict, u_l_predict, name_case)

        sxb1, sxc1, LBUB1, time1 = optimization.c043_SP(epsilon, LargeNumber, Tolerance, TimeLimitFC, TimeLimitSP, MaxIter, EPS, mpc, coefficients, u_data_train, b_display_SP)
        validation_cost = optimization.c046_evaluate(u_data_validation, coefficients, sxb1, sxc1, LBUB1)
        test_cost = optimization.c046_evaluate(u_data_test, coefficients, sxb1, sxc1, LBUB1)

        print('Calculated bound: {}'.format(LBUB1[-1, 0]))
        print('Validation bound:')
        print(validation_cost[:3])

        return validation_cost, test_cost, sxb1, sxc1, LBUB1, time1

    @staticmethod
    def c043_CCG_n1_faster(LargeNumber, Tolerance, TimeLimitFC, TimeLimitSP, MaxIter, EPS, mpc, coefficients, u_data_train, b_display_SP):
        """
        CCG with the ellipsoid uncertainty set
        """
        ## Time
        time_start = time.time()

        ## Load case
        case = Case()

        ## Load compact form
        Adexb = coefficients['Adexb']
        Adexc = coefficients['Adexc']
        Adixb = coefficients['Adixb']
        Adixc = coefficients['Adixc']
        Areu = coefficients['Areu']
        Arexc = coefficients['Arexc']
        Arey = coefficients['Arey']
        Arixc = coefficients['Arixc']
        Ariy = coefficients['Ariy']
        Bde = coefficients['Bde']
        Bdi = coefficients['Bdi']
        Bre = coefficients['Bre']
        Bri = coefficients['Bri']
        Cdxb = coefficients['Cdxb']
        Cdxc = coefficients['Cdxc']
        Cry = coefficients['Cry']

        try:
            ## Initiation
            # MP: Master problem
            MP = gp.Model('Master')
            xbMP = MP.addMVar((Cdxb.shape[0],), vtype=GRB.BINARY)
            xcMP = MP.addMVar((Cdxc.shape[0],), lb=-float('inf'), vtype=GRB.CONTINUOUS)
            yMP = MP.addMVar((MaxIter, Cry.shape[0]), lb=-float('inf'), vtype=GRB.CONTINUOUS)
            zetaMP = MP.addVar(lb=0, vtype=GRB.CONTINUOUS)
            MP.setObjective(Cdxb @ xbMP + Cdxc @ xcMP + zetaMP, GRB.MINIMIZE)
            MP.addConstr(Adexb @ xbMP + Adexc @ xcMP == Bde, name='de')
            MP.addConstr(Adixb @ xbMP + Adixc @ xcMP >= Bdi, name='di')
            MP.setParam('OutputFlag', 0) 

            # Feasibility check problem by mountain climbing
            # FC1: Fix uncertainty and optimize dual variables
            FC1 = gp.Model('Feasibility-Dual')
            muFC1 = FC1.addMVar((Bre.shape[0],), lb=-float('inf'), vtype=GRB.CONTINUOUS) # Dual variable of equation
            etaFC1 = FC1.addMVar((Bri.shape[0],), lb=-float('inf'), vtype=GRB.CONTINUOUS) # Dual variable of inequality
            FC1.addConstr(Arey.T @ muFC1 + Ariy.T @ etaFC1 == 0, name='de')
            FC1.addConstr(muFC1 <= 1, name='dru')
            FC1.addConstr(muFC1 >= -1, name='drl')
            FC1.addConstr(etaFC1 >= 0, name='di')
            FC1.setParam('OutputFlag', 0) 
            # FC2: Fix dual variables and optimize uncertainty
            FC2 = gp.Model('Feasibility-Uncertainty')
            uFC2 = FC2.addMVar((Areu.shape[1],), lb=-float('inf'), vtype=GRB.CONTINUOUS) # real u
            u_ldFC2 = FC2.addMVar((Areu.shape[1],), lb=-float('inf'), vtype=GRB.CONTINUOUS) # error - mu
            FC2.addConstr(u_ldFC2 == mpc['u_l_predict'] - uFC2 - mpc['error_mu'], name='u_e')
            FC2.addConstr(uFC2 >= mpc['u_ll'], name='u_lb')
            FC2.addConstr(mpc['u_lu'] >= uFC2, name='u_ub')
            FC2.addConstr(mpc['u_l_predict'] - uFC2 >= mpc['error_lb'], name='u_e_lb')
            FC2.addConstr(mpc['error_ub'] >= mpc['u_l_predict'] - uFC2, name='u_e_ub')
            FC2.addMQConstr(Q=mpc['error_sigma_inv'], c=None, sense='<', rhs=mpc['error_rho'], xQ_L=u_ldFC2, xQ_R=u_ldFC2, name='u_q')
            FC2.setParam('OutputFlag', 0) 

            ## Subproblem by mountain climbing
            # SP1: Fix uncertainty and optimize dual variables
            SP1 = gp.Model('Subproblem-Dual')
            muSP1 = SP1.addMVar((Bre.shape[0],), lb=-float('inf'), vtype=GRB.CONTINUOUS) # Dual variable of equation
            etaSP1 = SP1.addMVar((Bri.shape[0],), lb=-float('inf'), vtype=GRB.CONTINUOUS) # Dual variable of inequality
            SP1.addConstr(Arey.T @ muSP1 + Ariy.T @ etaSP1 == Cry, name='de')
            SP1.addConstr(etaSP1 >= 0, name='di')
            SP1.setParam('OutputFlag', 0) 
            # SP2: Fix dual variables and optimize uncertainty
            SP2 = gp.Model('Subproblem-Uncertainty')
            uSP2 = SP2.addMVar((Areu.shape[1],), lb=-float('inf'), vtype=GRB.CONTINUOUS)
            u_ldSP2 = SP2.addMVar((Areu.shape[1],), lb=-float('inf'), vtype=GRB.CONTINUOUS) # error - mu
            SP2.addConstr(u_ldSP2 == mpc['u_l_predict'] - uSP2 - mpc['error_mu'], name='u_e')
            SP2.addConstr(uSP2 >= mpc['u_ll'], name='u_lb')
            SP2.addConstr(mpc['u_lu'] >= uSP2, name='u_ub')
            SP2.addConstr(mpc['u_l_predict'] - uSP2 >= mpc['error_lb'], name='u_e_lb')
            SP2.addConstr(mpc['error_ub'] >= mpc['u_l_predict'] - uSP2, name='u_e_ub')
            SP2.addMQConstr(Q=mpc['error_sigma_inv'], c=None, sense='<', rhs=mpc['error_rho'], xQ_L=u_ldSP2, xQ_R=u_ldSP2, name='u_q')
            SP2.setParam('OutputFlag', 0) 

            # Other initiation
            LB = -float('inf') * np.ones((MaxIter,))
            UB = float('inf') * np.ones((MaxIter,))
            ulist = [] # The list of worst-case uncertainty
            Iter = 0

            ## Iteration
            while True:
                print('**************************************************************')
                print('Begin iteration: {}'.format(Iter))

                print('******************************MP******************************')
                MP.optimize()
                sxb = xbMP.X
                sxc = xcMP.X
                LB[Iter] = MP.ObjVal
                MPObjX = Cdxb @ sxb + Cdxc @ sxc

                print('LB: {} >= first-stage cost: {}'.format(LB[Iter], MPObjX))

                if (Iter > 0) & (LB[Iter] < LB[Iter - 1] * (1 + EPS)):
                    print('Cannot improve. Terminate.')
                    break

                # FC: Mountain climbing
                su = mpc['u_l_predict'] - mpc['error_mu'] # Initialize u
                for i in range(5):
                    FC1.addConstr((Bre - Areu @ su - Arexc @ sxc) @ muFC1 + (Bri - Arixc @ sxc) @ etaFC1 <= LargeNumber)
                    FC1.setObjective((Bre - Areu @ su - Arexc @ sxc) @ muFC1 + (Bri - Arixc @ sxc) @ etaFC1, GRB.MAXIMIZE)
                    FC1.optimize()
                    smu = muFC1.X
                    seta = etaFC1.X
                    FC2.setObjective((Bre - Areu @ uFC2 - Arexc @ sxc) @ smu + (Bri - Arixc @ sxc) @ seta, GRB.MAXIMIZE)
                    FC2.optimize()
                    su = uFC2.X

                print('test_u after mountain climbing FC')
                if not case.test_u(su, mpc, b_print=True, b_ellipsoid=True):
                    su = case.revise_u(su, mpc, EPS, b_print=True, b_ellipsoid=True)
                FCVal = (Bre - Areu @ su - Arexc @ sxc) @ smu + (Bri - Arixc @ sxc) @ seta

                print('FCObj: {}'.format(FCVal))
                
                if FCVal > EPS:
                    ulist.append(su)
                else:
                    # SP: Mountain climbing
                    su = mpc['u_l_predict'] - mpc['error_mu'] # Initialize u
                    for i in range(5):  
                        SP1.setObjective((Bre - Areu @ su - Arexc @ sxc) @ muSP1 + (Bri - Arixc @ sxc) @ etaSP1, GRB.MAXIMIZE)
                        SP1.optimize()
                        smu = muSP1.X
                        seta = etaSP1.X
                        SP2.setObjective((Bre - Areu @ uSP2 - Arexc @ sxc) @ smu + (Bri - Arixc @ sxc) @ seta, GRB.MAXIMIZE)
                        SP2.optimize()
                        su = uSP2.X

                    print('test_u after mountain climbing SP')
                    if not case.test_u(su, mpc, b_print=True, b_ellipsoid=True):
                        su = case.revise_u(su, mpc, EPS, b_print=True, b_ellipsoid=True)
                    SPVal = (Bre - Areu @ su - Arexc @ sxc) @ smu + (Bri - Arixc @ sxc) @ seta
                    UB[Iter] = MPObjX + SPVal

                    # Check convergence
                    print('UB: {}, LB: {}'.format(UB[Iter], LB[Iter]))
                    # if (np.min(UB) < float('inf')) & (np.min(UB) - LB[Iter] < Tolerance * np.min(UB)):
                    #     print('The algorithm converges.')
                    #     break
                    # else:
                    #     ulist.append(su)   
                    ulist.append(su)
                    
                # Add constraints to MP
                MP.addConstr(zetaMP >= Cry @ yMP[Iter, :])
                MP.addConstr(Areu @ ulist[-1] + Arexc @ xcMP + Arey @ yMP[Iter, :] == Bre)
                MP.addConstr(Arixc @ xcMP + Ariy @ yMP[Iter, :] >= Bri)
                
                Iter += 1
            
            # Output
            LBUB = np.concatenate((LB[:(Iter + 1)].reshape((-1, 1)), UB[:(Iter + 1)].reshape((-1, 1))), axis=1)

        except gp.GurobiError as e:
            print(f"Error code {e.errno}: {e}")

        except AttributeError:
            print("Encountered an attribute error")

        sobj = LBUB[-1, 0]

        ## Load u_data_train
        u_data = u_data_train
        num_data = u_data.shape[0]
        for i in range(num_data):
            try:
                ## Test feasibility of points
                fmodel = gp.Model('Feasibility')
                fs = fmodel.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS) # Slack
                fu = fmodel.addMVar((Areu.shape[1],), lb=-float('inf'), vtype=GRB.CONTINUOUS)
                fy = fmodel.addMVar((Cry.shape[0],), lb=-float('inf'), vtype=GRB.CONTINUOUS)
                fmodel.addConstr(u_data[i, :] - fu <= fs, name='s1')
                fmodel.addConstr(u_data[i, :] - fu >= - fs, name='s2')
                fmodel.addConstr(Areu @ fu + Arexc @ sxc + Arey @ fy == Bre, name='re')
                fmodel.addConstr(Ariy @ fy >= Bri - Arixc @ sxc, name='ri')
                fmodel.addConstr(sobj >= Cdxb @ sxb + Cdxc @ sxc + Cry @ fy, name='obj')
                fmodel.setObjective(fs, GRB.MINIMIZE)
                fmodel.setParam('OutputFlag', 0) 
                fmodel.optimize()

                if fs.X > 0:
                    ulist.append(u_data[i, :])
                    # Add constraints to MP
                    MP.addConstr(zetaMP >= Cry @ yMP[Iter, :])
                    MP.addConstr(Areu @ ulist[-1] + Arexc @ xcMP + Arey @ yMP[Iter, :] == Bre)
                    MP.addConstr(Arixc @ xcMP + Ariy @ yMP[Iter, :] >= Bri)

                    print('**************************************************************')
                    print('Begin iteration: {}'.format(Iter))
                    print('i/num_data: {}/{}'.format(i, num_data))

                    print('******************************MP******************************')
                    MP.optimize()
                    sxb = xbMP.X
                    sxc = xcMP.X
                    sobj = MP.ObjVal
                    LBUB = np.concatenate((LBUB, np.array([sobj, np.inf]).reshape((1, -1))), axis=0)

                    Iter += 1

            except gp.GurobiError as e:
                print(f"Error code {e.errno}: {e}")

            except AttributeError:
                print("Encountered an attribute error")

        time_elapsed = time.time() - time_start

        return ulist, sxb, sxc, LBUB, time_elapsed
    
    @staticmethod
    def c043_list(LargeNumber, Tolerance, TimeLimitFC, TimeLimitSP, MaxIter, EPS, mpc, coefficients, u_data_train_original, b_display_SP):
        """
        Using a list of data
        """
        ## Time
        time_start = time.time()

        ## Load case
        case = Case()

        ## Load compact form
        Adexb = coefficients['Adexb']
        Adexc = coefficients['Adexc']
        Adixb = coefficients['Adixb']
        Adixc = coefficients['Adixc']
        Areu = coefficients['Areu']
        Arexc = coefficients['Arexc']
        Arey = coefficients['Arey']
        Arixc = coefficients['Arixc']
        Ariy = coefficients['Ariy']
        Bde = coefficients['Bde']
        Bdi = coefficients['Bdi']
        Bre = coefficients['Bre']
        Bri = coefficients['Bri']
        Cdxb = coefficients['Cdxb']
        Cdxc = coefficients['Cdxc']
        Cry = coefficients['Cry']

        ## Initiation
        LB = -float('inf') * np.ones((MaxIter,))
        Iter = 0
        ## Load u_data_train
        u_data = u_data_train_original
        num_data = u_data.shape[0]
        MaxIter = num_data
        
        # MP: Master problem
        MP = gp.Model('Master')
        xbMP = MP.addMVar((Cdxb.shape[0],), vtype=GRB.BINARY)
        xcMP = MP.addMVar((Cdxc.shape[0],), lb=-float('inf'), vtype=GRB.CONTINUOUS)
        yMP = MP.addMVar((MaxIter, Cry.shape[0]), lb=-float('inf'), vtype=GRB.CONTINUOUS)
        zetaMP = MP.addVar(lb=0, vtype=GRB.CONTINUOUS)
        MP.setObjective(Cdxb @ xbMP + Cdxc @ xcMP + zetaMP, GRB.MINIMIZE)
        MP.addConstr(Adexb @ xbMP + Adexc @ xcMP == Bde, name='de')
        MP.addConstr(Adixb @ xbMP + Adixc @ xcMP >= Bdi, name='di')
        
        MP.setParam('OutputFlag', 0) 
        print('**************************************************************')
        print('Begin iteration: {}'.format(Iter))
        print('******************************MP******************************')
        MP.addConstr(zetaMP >= Cry @ yMP[Iter, :])
        MP.addConstr(Areu @ u_data[0, :] + Arexc @ xcMP + Arey @ yMP[Iter, :] == Bre)
        MP.addConstr(Arixc @ xcMP + Ariy @ yMP[Iter, :] >= Bri)
        MP.optimize()
        sxb = xbMP.X
        sxc = xcMP.X
        sobj = MP.ObjVal
        LB[Iter] = MP.ObjVal
        Iter = Iter + 1    
        print('LB: {}'.format(LB[Iter]))
        
        for i in range(num_data):
            try:
                ## Test feasibility of points
                fmodel = gp.Model('Feasibility')
                fs = fmodel.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS) # Slack
                fu = fmodel.addMVar((Areu.shape[1],), lb=-float('inf'), vtype=GRB.CONTINUOUS)
                fy = fmodel.addMVar((Cry.shape[0],), lb=-float('inf'), vtype=GRB.CONTINUOUS)
                fmodel.addConstr(u_data[i, :] - fu <= fs, name='s1')
                fmodel.addConstr(u_data[i, :] - fu >= - fs, name='s2')
                fmodel.addConstr(Areu @ fu + Arexc @ sxc + Arey @ fy == Bre, name='re')
                fmodel.addConstr(Ariy @ fy >= Bri - Arixc @ sxc, name='ri')
                fmodel.addConstr(sobj >= Cdxb @ sxb + Cdxc @ sxc + Cry @ fy, name='obj')
                fmodel.setObjective(fs, GRB.MINIMIZE)
                fmodel.setParam('OutputFlag', 0) 
                fmodel.optimize()

                print(fs.X)

                if fs.X > 0.02:

                    # Add constraints to MP
                    MP.addConstr(zetaMP >= Cry @ yMP[Iter, :])
                    MP.addConstr(Areu @ u_data[i, :] + Arexc @ xcMP + Arey @ yMP[Iter, :] == Bre)
                    MP.addConstr(Arixc @ xcMP + Ariy @ yMP[Iter, :] >= Bri)

                    print('**************************************************************')
                    print('Begin iteration: {}'.format(Iter))
                    print('i/num_data: {}/{}'.format(i, num_data))

                    print('******************************MP******************************')
                    MP.optimize()
                    sxb = xbMP.X
                    sxc = xcMP.X
                    sobj = MP.ObjVal
                    LB[Iter] = MP.ObjVal
                    print('LB: {}'.format(LB[Iter]))

                    Iter += 1

            except gp.GurobiError as e:
                print(f"Error code {e.errno}: {e}")

            except AttributeError:
                print("Encountered an attribute error")

        time_elapsed = time.time() - time_start
        LB = np.concatenate((LB.reshape(-1, 1), LB.reshape(-1, 1)), axis=1)
        LB = LB[:Iter]

        return sxb, sxc, LB, time_elapsed
    
    @staticmethod
    def c043_list_approx(epsilon, LargeNumber, Tolerance, TimeLimitFC, TimeLimitSP, MaxIter, EPS, mpc, coefficients, u_data_train_original, b_display_SP):
        """
        Using a portion of the list of data
        """
        ## Time
        time_start = time.time()

        ## Load case
        case = Case()

        ## Load compact form
        Adexb = coefficients['Adexb']
        Adexc = coefficients['Adexc']
        Adixb = coefficients['Adixb']
        Adixc = coefficients['Adixc']
        Areu = coefficients['Areu']
        Arexc = coefficients['Arexc']
        Arey = coefficients['Arey']
        Arixc = coefficients['Arixc']
        Ariy = coefficients['Ariy']
        Bde = coefficients['Bde']
        Bdi = coefficients['Bdi']
        Bre = coefficients['Bre']
        Bri = coefficients['Bri']
        Cdxb = coefficients['Cdxb']
        Cdxc = coefficients['Cdxc']
        Cry = coefficients['Cry']

        ## Initiation
        LB = -float('inf') * np.ones((MaxIter,))
        Iter = 0
        ## Load u_data_train
        u_data = u_data_train_original
        num_data = u_data.shape[0]
        rank = np.ceil((1 - epsilon) * num_data).astype(int)
        
        # MP: Master problem
        MP = gp.Model('Master')
        xbMP = MP.addMVar((Cdxb.shape[0],), vtype=GRB.BINARY)
        xcMP = MP.addMVar((Cdxc.shape[0],), lb=-float('inf'), vtype=GRB.CONTINUOUS)
        yMP = MP.addMVar((MaxIter, Cry.shape[0]), lb=-float('inf'), vtype=GRB.CONTINUOUS)
        zetaMP = MP.addVar(lb=0, vtype=GRB.CONTINUOUS)
        MP.setObjective(Cdxb @ xbMP + Cdxc @ xcMP + zetaMP, GRB.MINIMIZE)
        MP.addConstr(Adexb @ xbMP + Adexc @ xcMP == Bde, name='de')
        MP.addConstr(Adixb @ xbMP + Adixc @ xcMP >= Bdi, name='di')
        MP.setParam('OutputFlag', 0) 
        print('**************************************************************')
        print('Begin iteration: {}'.format(Iter))
        print('******************************MP******************************')
        MP.addConstr(zetaMP >= Cry @ yMP[Iter, :])
        MP.addConstr(Areu @ u_data[0, :] + Arexc @ xcMP + Arey @ yMP[Iter, :] == Bre)
        MP.addConstr(Arixc @ xcMP + Ariy @ yMP[Iter, :] >= Bri)
        MP.optimize()
        sxb = xbMP.X
        sxc = xcMP.X
        sobj = MP.ObjVal
        LB[Iter] = MP.ObjVal
        Iter = Iter + 1    
        print('LB: {}'.format(LB[Iter]))
        
        while Iter < MaxIter:
            try:
                testvalue = np.zeros((num_data,))
                for n in range(num_data):
                    ## Test feasibility of points
                    fmodel = gp.Model('Feasibility')
                    fs = fmodel.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS) # Slack
                    fu = fmodel.addMVar((Areu.shape[1],), lb=-float('inf'), vtype=GRB.CONTINUOUS)
                    fy = fmodel.addMVar((Cry.shape[0],), lb=-float('inf'), vtype=GRB.CONTINUOUS)
                    fmodel.addConstr(u_data[n, :] - fu <= fs, name='s1')
                    fmodel.addConstr(u_data[n, :] - fu >= - fs, name='s2')
                    fmodel.addConstr(Areu @ fu + Arexc @ sxc + Arey @ fy == Bre, name='re')
                    fmodel.addConstr(Ariy @ fy >= Bri - Arixc @ sxc, name='ri')
                    fmodel.addConstr(sobj >= Cdxb @ sxb + Cdxc @ sxc + Cry @ fy, name='obj')
                    fmodel.setObjective(fs, GRB.MINIMIZE)
                    fmodel.setParam('OutputFlag', 0) 
                    fmodel.optimize()
                    testvalue[n] = fs.X

                index_u_data = np.argsort(testvalue)[rank - 1]
                print(testvalue[index_u_data])

                if testvalue[index_u_data] < 1e-5:
                    break
                else:

                    # Add constraints to MP
                    MP.addConstr(zetaMP >= Cry @ yMP[Iter, :])
                    MP.addConstr(Areu @ u_data[index_u_data, :] + Arexc @ xcMP + Arey @ yMP[Iter, :] == Bre)
                    MP.addConstr(Arixc @ xcMP + Ariy @ yMP[Iter, :] >= Bri)

                    print('**************************************************************')
                    print('Begin iteration: {}'.format(Iter))

                    print('******************************MP******************************')
                    MP.optimize()
                    sxb = xbMP.X
                    sxc = xcMP.X
                    sobj = MP.ObjVal
                    LB[Iter] = MP.ObjVal
                    print('LB: {}'.format(LB[Iter]))

                    Iter += 1

            except gp.GurobiError as e:
                print(f"Error code {e.errno}: {e}")

            except AttributeError:
                print("Encountered an attribute error")

        time_elapsed = time.time() - time_start
        LB = np.concatenate((LB.reshape(-1, 1), LB.reshape(-1, 1)), axis=1)
        LB = LB[:Iter]

        return sxb, sxc, LB, time_elapsed
    
    @staticmethod
    def c043_SP(epsilon, LargeNumber, Tolerance, TimeLimitFC, TimeLimitSP, MaxIter, EPS, mpc, coefficients, u_data_train_original, b_display_SP):
        """
        Using a portion of data
        """
        ## Time
        time_start = time.time()

        ## Load case
        case = Case()

        ## Load compact form
        Adexb = coefficients['Adexb']
        Adexc = coefficients['Adexc']
        Adixb = coefficients['Adixb']
        Adixc = coefficients['Adixc']
        Areu = coefficients['Areu']
        Arexc = coefficients['Arexc']
        Arey = coefficients['Arey']
        Arixc = coefficients['Arixc']
        Ariy = coefficients['Ariy']
        Bde = coefficients['Bde']
        Bdi = coefficients['Bdi']
        Bre = coefficients['Bre']
        Bri = coefficients['Bri']
        Cdxb = coefficients['Cdxb']
        Cdxc = coefficients['Cdxc']
        Cry = coefficients['Cry']

        ## Initiation
        LB = -float('inf') * np.ones((MaxIter,))
        Iter = 0
        ## Load u_data_train
        u_data = u_data_train_original
        num_data = u_data.shape[0]
        
        # MP: Master problem
        MP = gp.Model('Master')
        xbMP = MP.addMVar((Cdxb.shape[0],), vtype=GRB.BINARY)
        xcMP = MP.addMVar((Cdxc.shape[0],), lb=-float('inf'), vtype=GRB.CONTINUOUS)
        yMP = MP.addMVar((num_data, Cry.shape[0]), lb=-float('inf'), vtype=GRB.CONTINUOUS)
        zetaMP = MP.addVar(lb=0, vtype=GRB.CONTINUOUS)
        fs = MP.addMVar((num_data,), lb=-float('inf'), vtype=GRB.CONTINUOUS)
        fu = MP.addMVar((num_data, Areu.shape[1]), lb=-float('inf'), vtype=GRB.CONTINUOUS)
        zMP = MP.addMVar((num_data,), vtype=GRB.BINARY)
        MP.setObjective(Cdxb @ xbMP + Cdxc @ xcMP + zetaMP, GRB.MINIMIZE)
        MP.addConstr(Adexb @ xbMP + Adexc @ xcMP == Bde, name='de')
        MP.addConstr(Adixb @ xbMP + Adixc @ xcMP >= Bdi, name='di')
        MP.addConstr(gp.quicksum(zMP) <= num_data * epsilon)
        for i in range(num_data): # Add uncertainty data as initiation
            MP.addConstr(u_data[i, :] - fu[i, :] <= fs[i], name='s1')
            MP.addConstr(u_data[i, :] - fu[i, :] >= - fs[i], name='s2')
            MP.addConstr(zetaMP >= Cry @ yMP[i, :], name='rcd')
            MP.addConstr(Areu @ fu[i, :] + Arexc @ xcMP + Arey @ yMP[i, :] == Bre, name='red')
            MP.addConstr(Arixc @ xcMP + Ariy @ yMP[i, :] >= Bri, name='rid')
            MP.addConstr(fs[i] <= LargeNumber * zMP[i])
        MP.setParam('OutputFlag', 1) 
        
        MP.optimize()
        sxb = xbMP.X
        sxc = xcMP.X
        sobj = MP.ObjVal
        LB = np.array([sobj, sobj]).reshape((1, 2))
        
        time_elapsed = time.time() - time_start

        return sxb, sxc, LB, time_elapsed
    
    @staticmethod
    def c044_reconstruction_faster(epsilon, delta, coefficients, u_data_train_n2, sxb, sxc, LBUB):
        """
        Uncertainty set reconstruction
        """

        print('(((((((((((((((((((((((((((((c044)))))))))))))))))))))))))))))')

        ## Load compact form
        Areu = coefficients['Areu']
        Arexc = coefficients['Arexc']
        Arey = coefficients['Arey']
        Arixc = coefficients['Arixc']
        Ariy = coefficients['Ariy']
        Bre = coefficients['Bre']
        Bri = coefficients['Bri']
        Cdxb = coefficients['Cdxb']
        Cdxc = coefficients['Cdxc']
        Cry = coefficients['Cry']

        ## Load u_data_train_n2
        u_data = u_data_train_n2
        num_data = u_data.shape[0]

        ## Load CCG solution
        sobj = LBUB[-1, 0]

        sfs = np.zeros((num_data,))
        for i in range(num_data):
            try:
                ## Test feasibility of points
                fmodel = gp.Model('Feasibility')
                fs = fmodel.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS) # Slack
                fu = fmodel.addMVar((Areu.shape[1],), lb=-float('inf'), vtype=GRB.CONTINUOUS)
                fy = fmodel.addMVar((Cry.shape[0],), lb=-float('inf'), vtype=GRB.CONTINUOUS)
                fmodel.addConstr(u_data[i, :] - fu <= fs, name='s1')
                fmodel.addConstr(u_data[i, :] - fu >= - fs, name='s2')
                fmodel.addConstr(Areu @ fu + Arexc @ sxc + Arey @ fy == Bre, name='re')
                fmodel.addConstr(Ariy @ fy >= Bri - Arixc @ sxc, name='ri')
                fmodel.addConstr(sobj >= Cdxb @ sxb + Cdxc @ sxc + Cry @ fy, name='obj')
                fmodel.setObjective(fs, GRB.MINIMIZE)
                fmodel.setParam('OutputFlag', 0) 
                fmodel.optimize()

                sfs[i] = fs.X

            except gp.GurobiError as e:
                print(f"Error code {e.errno}: {e}")

            except AttributeError:
                print("Encountered an attribute error")

        print('There are {} data in n2, where {} of them are not feasible.'.format(num_data, np.count_nonzero(sfs)))
        u_data = u_data[sfs == 0]
        num_data = u_data.shape[0]

        sos = np.zeros((num_data,))
        for i in range(num_data):
            try:
                ## Calculate objective for all points
                omodel = gp.Model('Objective')
                os = omodel.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS) # Slack
                oy = omodel.addMVar((Cry.shape[0],), lb=-float('inf'), vtype=GRB.CONTINUOUS)
                omodel.addConstr(Areu @ u_data[i, :] + Arexc @ sxc + Arey @ oy == Bre, name='re')
                omodel.addConstr(Ariy @ oy >= Bri - Arixc @ sxc, name='ri')
                omodel.addConstr(sobj + os >= Cdxb @ sxb + Cdxc @ sxc + Cry @ oy, name='obj')
                omodel.setObjective(os, GRB.MINIMIZE)
                omodel.setParam('OutputFlag', 0) 
                omodel.optimize()

                sos[i] = os.X

            except gp.GurobiError as e:
                print(f"Error code {e.errno}: {e}")

            except AttributeError:
                print("Encountered an attribute error")

        rank = CombHandler().get_rank(num_data, epsilon, delta)
        radius = sos[np.argsort(sos)[rank - 1]]
        u_data = u_data[np.argsort(sos)[:rank]]
        
        try:
            ## Form the model of the uncertainty set
            umodel = gp.Model('Uncertainty')
            uu = umodel.addMVar((Areu.shape[1],), lb=-float('inf'), vtype=GRB.CONTINUOUS)
            umodel.update()
            numuu = umodel.NumVars
            uy = umodel.addMVar((Cry.shape[0],), lb=-float('inf'), vtype=GRB.CONTINUOUS)
            umodel.addConstr(Areu @ uu + Arexc @ sxc + Arey @ uy == Bre, name='re')
            umodel.update()
            numue = umodel.NumConstrs
            umodel.addConstr(Ariy @ uy >= Bri - Arixc @ sxc, name='ri')
            umodel.addConstr(sobj + radius >= Cdxb @ sxb + Cdxc @ sxc + Cry @ uy, name='obj')
            umodel.setObjective(0, GRB.MINIMIZE)
            umodel.setParam('OutputFlag', 0) 
            umodel.optimize()

            A = umodel.getA()
            B = np.array(umodel.getAttr('RHS', umodel.getConstrs()))
            sense = umodel.getAttr('Sense', umodel.getConstrs())
            for i, x in enumerate(sense):
                if x == '<':
                    A[i, :] = - A[i, :]
                    B[i] = - B[i]

            Aueu = A[:numue, :numuu]
            Auey = A[:numue, numuu:]
            Bue = B[:numue]
            Auiy = A[numue:, numuu:]
            Bui = B[numue:]

            coefficients['Aueu'] = Aueu
            coefficients['Auey'] = Auey
            coefficients['Auiy'] = Auiy
            coefficients['Bue'] = Bue
            coefficients['Bui'] = Bui

        except gp.GurobiError as e:
            print(f"Error code {e.errno}: {e}")

        except AttributeError:
            print("Encountered an attribute error")

        return radius, u_data, coefficients
    
    @staticmethod
    def c045_CCG_n2_faster(LargeNumber, Tolerance, TimeLimitFC, TimeLimitSP, MaxIter, EPS, mpc, coefficients, u_data):
        """
        CCG with the reconstructed uncertainty set
        """
        print('(((((((((((((((((((((((((((((c045)))))))))))))))))))))))))))))')

        ## Time
        time_start = time.time()

        ## Load case
        case = Case()

        ## Load compact form
        Adexb = coefficients['Adexb']
        Adexc = coefficients['Adexc']
        Adixb = coefficients['Adixb']
        Adixc = coefficients['Adixc']
        Areu = coefficients['Areu']
        Arexc = coefficients['Arexc']
        Arey = coefficients['Arey']
        Arixc = coefficients['Arixc']
        Ariy = coefficients['Ariy']
        Aueu = coefficients['Aueu']
        Auey = coefficients['Auey']
        Auiy = coefficients['Auiy']
        Bde = coefficients['Bde']
        Bdi = coefficients['Bdi']
        Bre = coefficients['Bre']
        Bri = coefficients['Bri']
        Bue = coefficients['Bue']
        Bui = coefficients['Bui']   
        Cdxb = coefficients['Cdxb']
        Cdxc = coefficients['Cdxc']
        Cry = coefficients['Cry']

        try:
            ## Initiation
            # MP: Master problem
            MP = gp.Model('Master')
            xbMP = MP.addMVar((Cdxb.shape[0],), vtype=GRB.BINARY)
            xcMP = MP.addMVar((Cdxc.shape[0],), lb=-float('inf'), vtype=GRB.CONTINUOUS)
            yMP = MP.addMVar((MaxIter, Cry.shape[0]), lb=-float('inf'), vtype=GRB.CONTINUOUS)
            zetaMP = MP.addVar(lb=0, vtype=GRB.CONTINUOUS)
            MP.setObjective(Cdxb @ xbMP + Cdxc @ xcMP + zetaMP, GRB.MINIMIZE)
            MP.addConstr(Adexb @ xbMP + Adexc @ xcMP == Bde, name='de')
            MP.addConstr(Adixb @ xbMP + Adixc @ xcMP >= Bdi, name='di')
            MP.setParam('OutputFlag', 0) 

            # Feasibility check problem by mountain climbing
            # FC1: Fix uncertainty and optimize dual variables
            FC1 = gp.Model('Feasibility-Dual')
            muFC1 = FC1.addMVar((Bre.shape[0],), lb=-float('inf'), vtype=GRB.CONTINUOUS) # Dual variable of equation
            etaFC1 = FC1.addMVar((Bri.shape[0],), lb=-float('inf'), vtype=GRB.CONTINUOUS) # Dual variable of inequality
            FC1.addConstr(Arey.T @ muFC1 + Ariy.T @ etaFC1 == 0, name='de')
            FC1.addConstr(muFC1 <= 1, name='dru')
            FC1.addConstr(muFC1 >= -1, name='drl')
            FC1.addConstr(etaFC1 >= 0, name='di')
            FC1.setParam('OutputFlag', 0) 
            # FC2: Fix dual variables and optimize uncertainty
            FC2 = gp.Model('Feasibility-Uncertainty')
            uFC2 = FC2.addMVar((Areu.shape[1],), lb=-float('inf'), vtype=GRB.CONTINUOUS)
            yFC2 = FC2.addMVar((Auey.shape[1],), lb=-float('inf'), vtype=GRB.CONTINUOUS) # Second-stage variable for defining the uncertainty set
            FC2.addConstr(uFC2 >= mpc['u_ll'], name='u_lb')
            FC2.addConstr(mpc['u_lu'] >= uFC2, name='u_ub')
            FC2.addConstr(mpc['u_l_predict'] - uFC2 >= mpc['error_lb'], name='u_e_lb')
            FC2.addConstr(mpc['error_ub'] >= mpc['u_l_predict'] - uFC2, name='u_e_ub')
            FC2.addConstr(Aueu @ uFC2 + Auey @ yFC2 == Bue, 'ue')
            FC2.addConstr(Auiy @ yFC2 >= Bui, 'ui')
            FC2.setParam('OutputFlag', 0) 

            ## Subproblem by mountain climbing
            # SP1: Fix uncertainty and optimize dual variables
            SP1 = gp.Model('Subproblem-Dual')
            muSP1 = SP1.addMVar((Bre.shape[0],), lb=-float('inf'), vtype=GRB.CONTINUOUS) # Dual variable of equation
            etaSP1 = SP1.addMVar((Bri.shape[0],), lb=-float('inf'), vtype=GRB.CONTINUOUS) # Dual variable of inequality
            SP1.addConstr(Arey.T @ muSP1 + Ariy.T @ etaSP1 == Cry, name='de')
            SP1.addConstr(etaSP1 >= 0, name='di')
            SP1.setParam('OutputFlag', 0) 
            # SP2: Fix dual variables and optimize uncertainty
            SP2 = gp.Model('Subproblem-Uncertainty')
            uSP2 = SP2.addMVar((Areu.shape[1],), lb=-float('inf'), vtype=GRB.CONTINUOUS)
            ySP2 = SP2.addMVar((Auey.shape[1],), lb=-float('inf'), vtype=GRB.CONTINUOUS) # Second-stage variable for defining the uncertainty set
            SP2.addConstr(uSP2 >= mpc['u_ll'], name='u_lb')
            SP2.addConstr(mpc['u_lu'] >= uSP2, name='u_ub')
            SP2.addConstr(mpc['u_l_predict'] - uSP2 >= mpc['error_lb'], name='u_e_lb')
            SP2.addConstr(mpc['error_ub'] >= mpc['u_l_predict'] - uSP2, name='u_e_ub')
            SP2.addConstr(Aueu @ uSP2 + Auey @ ySP2 == Bue, 'ue')
            SP2.addConstr(Auiy @ ySP2 >= Bui, 'ui')
            SP2.setParam('OutputFlag', 0) 

            # Other initiation
            LB = -float('inf') * np.ones((MaxIter,))
            UB = float('inf') * np.ones((MaxIter,)) # Upper bound
            ulist = [] # The list of worst-case uncertainty
            Iter = 0

            ## Iteration
            while True:
                print('**************************************************************')
                print('Begin iteration: {}'.format(Iter))

                print('******************************MP******************************')
                MP.optimize()
                sxb = xbMP.X
                sxc = xcMP.X
                LB[Iter] = MP.ObjVal
                MPObjX = Cdxb @ sxb + Cdxc @ sxc

                print('LB: {} >= first-stage cost: {}'.format(LB[Iter], MPObjX))

                if (Iter > 0) & (LB[Iter] < LB[Iter - 1] * (1 + EPS)):
                    print('Cannot improve. Terminate.')
                    break

                # FC: Mountain climbing
                su = mpc['u_l_predict'] - mpc['error_mu'] # Initialize u
                for i in range(5):
                    FC1.addConstr((Bre - Areu @ su - Arexc @ sxc) @ muFC1 + (Bri - Arixc @ sxc) @ etaFC1 <= LargeNumber)
                    FC1.setObjective((Bre - Areu @ su - Arexc @ sxc) @ muFC1 + (Bri - Arixc @ sxc) @ etaFC1, GRB.MAXIMIZE)
                    FC1.optimize()
                    smu = muFC1.X
                    seta = etaFC1.X
                    FC2.setObjective((Bre - Areu @ uFC2 - Arexc @ sxc) @ smu + (Bri - Arixc @ sxc) @ seta, GRB.MAXIMIZE)
                    FC2.optimize()
                    su = uFC2.X

                print('test_u after mountain climbing FC')
                if not case.test_u(su, mpc, b_print=True, b_ellipsoid=False):
                    su = case.revise_u(su, mpc, EPS, b_print=True, b_ellipsoid=False)
                FCVal = (Bre - Areu @ su - Arexc @ sxc) @ smu + (Bri - Arixc @ sxc) @ seta

                print('FCObj (revised to be feasible): {}'.format(FCVal))
                
                if FCVal > EPS:
                    ulist.append(su)
                else:
                    # SP: Mountain climbing
                    su = mpc['u_l_predict'] - mpc['error_mu'] # Initialize u
                    for i in range(5):  
                        SP1.setObjective((Bre - Areu @ su - Arexc @ sxc) @ muSP1 + (Bri - Arixc @ sxc) @ etaSP1, GRB.MAXIMIZE)
                        SP1.optimize()
                        smu = muSP1.X
                        seta = etaSP1.X
                        SP2.setObjective((Bre - Areu @ uSP2 - Arexc @ sxc) @ smu + (Bri - Arixc @ sxc) @ seta, GRB.MAXIMIZE)
                        SP2.optimize()
                        su = uSP2.X

                    print('test_u after mountain climbing SP')
                    if not case.test_u(su, mpc, b_print=True, b_ellipsoid=False):
                        su = case.revise_u(su, mpc, EPS, b_print=True, b_ellipsoid=False)
                    SPVal = (Bre - Areu @ su - Arexc @ sxc) @ smu + (Bri - Arixc @ sxc) @ seta
                    
                    UB[Iter] = MPObjX + SPVal

                    # Check convergence
                    print('UB: {}, LB: {}'.format(UB[Iter], LB[Iter]))
                    ulist.append(su)       
                    
                # Add constraints to MP
                MP.addConstr(zetaMP >= Cry @ yMP[Iter, :])
                MP.addConstr(Areu @ ulist[-1] + Arexc @ xcMP + Arey @ yMP[Iter, :] == Bre)
                MP.addConstr(Arixc @ xcMP + Ariy @ yMP[Iter, :] >= Bri)
                
                Iter += 1
            
            # Output
            LBUB = np.concatenate((LB[:(Iter + 1)].reshape((-1, 1)), UB[:(Iter + 1)].reshape((-1, 1))), axis=1)

        except gp.GurobiError as e:
            print(f"Error code {e.errno}: {e}")

        except AttributeError:
            print("Encountered an attribute error")

        sobj = LBUB[-1, 0]

        ## Load u_data_train
        num_data = u_data.shape[0]
        for i in range(num_data):
            try:
                ## Test feasibility of points
                fmodel = gp.Model('Feasibility')
                fs = fmodel.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS) # Slack
                fu = fmodel.addMVar((Areu.shape[1],), lb=-float('inf'), vtype=GRB.CONTINUOUS)
                fy = fmodel.addMVar((Cry.shape[0],), lb=-float('inf'), vtype=GRB.CONTINUOUS)
                fmodel.addConstr(u_data[i, :] - fu <= fs, name='s1')
                fmodel.addConstr(u_data[i, :] - fu >= - fs, name='s2')
                fmodel.addConstr(Areu @ fu + Arexc @ sxc + Arey @ fy == Bre, name='re')
                fmodel.addConstr(Ariy @ fy >= Bri - Arixc @ sxc, name='ri')
                fmodel.addConstr(sobj >= Cdxb @ sxb + Cdxc @ sxc + Cry @ fy, name='obj')
                fmodel.setObjective(fs, GRB.MINIMIZE)
                fmodel.setParam('OutputFlag', 0) 
                fmodel.optimize()

                if fs.X > 0:
                    ulist.append(u_data[i, :])
                    # Add constraints to MP
                    MP.addConstr(zetaMP >= Cry @ yMP[Iter, :])
                    MP.addConstr(Areu @ ulist[-1] + Arexc @ xcMP + Arey @ yMP[Iter, :] == Bre)
                    MP.addConstr(Arixc @ xcMP + Ariy @ yMP[Iter, :] >= Bri)

                    print('**************************************************************')
                    print('Begin iteration: {}'.format(Iter))
                    print('i/num_data: {}/{}'.format(i, num_data))

                    print('******************************MP******************************')
                    MP.optimize()
                    sxb = xbMP.X
                    sxc = xcMP.X
                    sobj = MP.ObjVal
                    LBUB = np.concatenate((LBUB, np.array([sobj, np.inf]).reshape((1, -1))), axis=0)

                    Iter += 1

            except gp.GurobiError as e:
                print(f"Error code {e.errno}: {e}")

            except AttributeError:
                print("Encountered an attribute error")

        time_elapsed = time.time() - time_start
        interpret = {}

        return u_data, ulist, sxb, sxc, LBUB, time_elapsed, interpret
    
    @staticmethod
    def c046_evaluate_faster(u_data, coefficients, sxb, sxc, LBUB):
        """
        Use validation/test dataset to evaluate the obtained strategy
        """
        print('(((((((((((((((((((((((((((((c046)))))))))))))))))))))))))))))')

        ## Load compact form
        Areu = coefficients['Areu']
        Arexc = coefficients['Arexc']
        Arey = coefficients['Arey']
        Arixc = coefficients['Arixc']
        Ariy = coefficients['Ariy']
        Bre = coefficients['Bre']
        Bri = coefficients['Bri']
        Cdxb = coefficients['Cdxb']
        Cdxc = coefficients['Cdxc']
        Cry = coefficients['Cry']

        num_data = u_data.shape[0]

        ## Load CCG solution
        sobj = LBUB[-1, 0]

        ## Cost vector to indicate infeasibility/objective
        cost = np.ones((num_data,)) * float('inf')

        sfs = np.zeros((num_data,))
        for i in range(num_data):
            try:
                ## Test feasibility of points
                fmodel = gp.Model('Feasibility')
                fs = fmodel.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS) # Slack
                fu = fmodel.addMVar((Areu.shape[1],), lb=-float('inf'), vtype=GRB.CONTINUOUS)
                fy = fmodel.addMVar((Cry.shape[0],), lb=-float('inf'), vtype=GRB.CONTINUOUS)
                fmodel.addConstr(u_data[i, :] - fu <= fs, name='s1')
                fmodel.addConstr(u_data[i, :] - fu >= - fs, name='s2')
                fmodel.addConstr(Areu @ fu + Arexc @ sxc + Arey @ fy == Bre, name='re')
                fmodel.addConstr(Ariy @ fy >= Bri - Arixc @ sxc, name='ri')
                fmodel.addConstr(sobj >= Cdxb @ sxb + Cdxc @ sxc + Cry @ fy, name='obj')
                fmodel.setObjective(fs, GRB.MINIMIZE)
                fmodel.setParam('OutputFlag', 0) 
                fmodel.optimize()

                sfs[i] = fs.X

            except gp.GurobiError as e:
                print(f"Error code {e.errno}: {e}")

            except AttributeError:
                print("Encountered an attribute error")

        print('There are {} out of {} data that are not feasible.'.format(np.count_nonzero(sfs), num_data))
        u_data = u_data[sfs == 0]
        num_data = u_data.shape[0]

        for i in range(num_data):
            try:
                ## Calculate objective for all points
                omodel = gp.Model('Objective')
                os = omodel.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS) # Slack
                oy = omodel.addMVar((Cry.shape[0],), lb=-float('inf'), vtype=GRB.CONTINUOUS)
                omodel.addConstr(Areu @ u_data[i, :] + Arexc @ sxc + Arey @ oy == Bre, name='re')
                omodel.addConstr(Ariy @ oy >= Bri - Arixc @ sxc, name='ri')
                omodel.addConstr(sobj + os >= Cdxb @ sxb + Cdxc @ sxc + Cry @ oy, name='obj')
                omodel.setObjective(os, GRB.MINIMIZE)
                omodel.setParam('OutputFlag', 0) 
                omodel.optimize()

                cost[i] = sobj + os.X

            except gp.GurobiError as e:
                print(f"Error code {e.errno}: {e}")

            except AttributeError:
                print("Encountered an attribute error")

        cost = np.sort(cost)[::-1]

        return cost
    
    @staticmethod
    def c046_evaluate_faster_order(u_data, coefficients, sxb, sxc, LBUB):
        """
        Use validation/test dataset to evaluate the obtained strategy
        """
        print('(((((((((((((((((((((((((((((c046)))))))))))))))))))))))))))))')

        ## Load compact form
        Areu = coefficients['Areu']
        Arexc = coefficients['Arexc']
        Arey = coefficients['Arey']
        Arixc = coefficients['Arixc']
        Ariy = coefficients['Ariy']
        Bre = coefficients['Bre']
        Bri = coefficients['Bri']
        Cdxb = coefficients['Cdxb']
        Cdxc = coefficients['Cdxc']
        Cry = coefficients['Cry']

        num_data = u_data.shape[0]

        ## Load CCG solution
        sobj = LBUB[-1, 0]

        ## Cost vector to indicate infeasibility/objective
        cost = np.zeros((num_data,))

        sfs = np.zeros((num_data,))
        for i in range(num_data):
            try:
                ## Test feasibility of points
                fmodel = gp.Model('Feasibility')
                fs = fmodel.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS) # Slack
                fu = fmodel.addMVar((Areu.shape[1],), lb=-float('inf'), vtype=GRB.CONTINUOUS)
                fy = fmodel.addMVar((Cry.shape[0],), lb=-float('inf'), vtype=GRB.CONTINUOUS)
                fmodel.addConstr(u_data[i, :] - fu <= fs, name='s1')
                fmodel.addConstr(u_data[i, :] - fu >= - fs, name='s2')
                fmodel.addConstr(Areu @ fu + Arexc @ sxc + Arey @ fy == Bre, name='re')
                fmodel.addConstr(Ariy @ fy >= Bri - Arixc @ sxc, name='ri')
                fmodel.addConstr(sobj >= Cdxb @ sxb + Cdxc @ sxc + Cry @ fy, name='obj')
                fmodel.setObjective(fs, GRB.MINIMIZE)
                fmodel.setParam('OutputFlag', 0) 
                fmodel.optimize()

                sfs[i] = fs.X

            except gp.GurobiError as e:
                print(f"Error code {e.errno}: {e}")

            except AttributeError:
                print("Encountered an attribute error")

        print('There are {} out of {} data that are not feasible.'.format(np.count_nonzero(sfs), num_data))
        cost[sfs > 0] = float('inf')

        for i in range(num_data):
            if sfs[i] > 0:
                continue
            try:
                ## Calculate objective for all points
                omodel = gp.Model('Objective')
                os = omodel.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS) # Slack
                oy = omodel.addMVar((Cry.shape[0],), lb=-float('inf'), vtype=GRB.CONTINUOUS)
                omodel.addConstr(Areu @ u_data[i, :] + Arexc @ sxc + Arey @ oy == Bre, name='re')
                omodel.addConstr(Ariy @ oy >= Bri - Arixc @ sxc, name='ri')
                omodel.addConstr(sobj + os >= Cdxb @ sxb + Cdxc @ sxc + Cry @ oy, name='obj')
                omodel.setObjective(os, GRB.MINIMIZE)
                omodel.setParam('OutputFlag', 0) 
                omodel.optimize()

                cost[i] = sobj + os.X

            except gp.GurobiError as e:
                print(f"Error code {e.errno}: {e}")

            except AttributeError:
                print("Encountered an attribute error")

        return cost