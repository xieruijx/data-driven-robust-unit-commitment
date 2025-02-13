import numpy as np
import time
import gurobipy as gp
from gurobipy import GRB

from utils.uncertainty import Uncertainty

class C043(object):
    """
    C043 class for CCG of ellipsoidal uncertainty set
    """

    @staticmethod
    def c043_CCG_n1(LargeNumber, Tolerance, TimeLimitFC, TimeLimitSP, MaxIter, EPS, mpc, coefficients, u_data_train, b_display_SP):
        """
        CCG with the ellipsoid uncertainty set
        """
        print('(((((((((((((((((((((((((((((c043)))))))))))))))))))))))))))))')
        ## Time
        time_start = time.time()

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
            while Iter < MaxIter:
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
                if not Uncertainty().test_u(su, mpc, b_print=True, b_ellipsoid=True):
                    su = Uncertainty().revise_u(su, mpc, EPS, b_print=True, b_ellipsoid=True)
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
                if Uncertainty().test_u(uFC.X, mpc, b_print=True, b_ellipsoid=True):
                    print('FC: su is in the uncertainty set.')
                    su = uFC.X
                    FCVal = FC.ObjVal
                else:
                    print('FC: su is not in the uncertainty set.')
                    su = Uncertainty().revise_u(uFC.X, mpc, EPS, b_print=True, b_ellipsoid=True) # Revise and mountain climbing
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
                    if not Uncertainty().test_u(su, mpc, b_print=True, b_ellipsoid=True):
                        su = Uncertainty().revise_u(su, mpc, EPS, b_print=True, b_ellipsoid=True)
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
                    if not Uncertainty().test_u(su, mpc, b_print=True, b_ellipsoid=True):
                        su = Uncertainty().revise_u(su, mpc, EPS, b_print=True, b_ellipsoid=True)
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
                    if Uncertainty().test_u(uSP.X, mpc, b_print=True, b_ellipsoid=True):
                        SPVal = SP.ObjVal
                    else:
                        su = Uncertainty().revise_u(uSP.X, mpc, EPS, b_print=True, b_ellipsoid=True)
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
                        if not Uncertainty().test_u(su, mpc, b_print=True, b_ellipsoid=True):
                            su = Uncertainty().revise_u(su, mpc, EPS, b_print=True, b_ellipsoid=True)
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

        return ulist, sxb, sxc, LBUB, time_elapsed, interpret
    
    @staticmethod
    def c043_CCG_n1_faster(LargeNumber, Tolerance, TimeLimitFC, TimeLimitSP, MaxIter, EPS, mpc, coefficients, u_data_train, b_display_SP):
        """
        CCG with the ellipsoid uncertainty set
        """
        ## Time
        time_start = time.time()

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
            while Iter < MaxIter:
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
                if not Uncertainty().test_u(su, mpc, b_print=True, b_ellipsoid=True):
                    su = Uncertainty().revise_u(su, mpc, EPS, b_print=True, b_ellipsoid=True)
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
                    if not Uncertainty().test_u(su, mpc, b_print=True, b_ellipsoid=True):
                        su = Uncertainty().revise_u(su, mpc, EPS, b_print=True, b_ellipsoid=True)
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
            if Iter >= MaxIter:
                break
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

        sy = yMP.X[np.argmax(yMP.X @ Cry), :]

        time_elapsed = time.time() - time_start
        interpret = {}
        interpret['x_og'] = sxb[:(mpc['n_t'] * mpc['n_g'])].reshape((mpc['n_t'], mpc['n_g']))
        interpret['x_pg'] = sxc[:(mpc['n_t'] * mpc['n_g'])].reshape((mpc['n_t'], mpc['n_g']))
        interpret['x_rp'] = sxc[(mpc['n_t'] * mpc['n_g']):(mpc['n_t'] * mpc['n_g'] * 2)].reshape((mpc['n_t'], mpc['n_g']))
        interpret['x_rn'] = sxc[(mpc['n_t'] * mpc['n_g'] * 2):].reshape((mpc['n_t'], mpc['n_g']))
        interpret['y_rp'] = sy[:(mpc['n_t'] * mpc['n_g'])].reshape((mpc['n_t'], mpc['n_g']))
        interpret['y_rn'] = sy[(mpc['n_t'] * mpc['n_g']):].reshape((mpc['n_t'], mpc['n_g']))

        return ulist, sxb, sxc, LBUB, time_elapsed, interpret
    