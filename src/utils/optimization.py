import numpy as np
import time
import gurobipy as gp
from gurobipy import GRB

from utils.combhandler import CombHandler
from utils.case import Case
from utils.c032_weight_calculation import C032
from utils.c041_uncertainty_initiation import C041
from utils.c042_dispatch_model import C042

class Optimization(object):
    """
    Optimization class for the optimization process
    """
    
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

        type_r = parameter['type_r']
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

        train_real, train_predict, train_n1_real, train_n1_predict, train_n2_real, train_n2_predict, validation_real, validation_predict, test_real, test_predict, error_bounds = C032().c032_calculate_weight(num_groups, weight)
        error_mu, error_sigma, error_rho = C041().c041_initial_uncertainty(type_r, horizon, epsilon, delta, u_select, train_n1_real, train_n1_predict, train_n2_real, train_n2_predict)
        u_l_predict = validation_predict[(index_u_l_predict * horizon):((index_u_l_predict + 1) * horizon)]
        mpc, coefficients, u_data_train, u_data_train_n2, u_data_validation, u_data_test, u_data_train_original = C042().c042_dispatch_model(u_select, error_mu, error_sigma, error_rho, error_bounds, EPS, train_real, train_predict, train_n2_real, train_n2_predict, validation_real, validation_predict, test_real, test_predict, u_l_predict, name_case)

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

        type_r = parameter['type_r']
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

        train_real, train_predict, train_n1_real, train_n1_predict, train_n2_real, train_n2_predict, validation_real, validation_predict, test_real, test_predict, error_bounds = C032().c032_calculate_weight(num_groups, weight)
        error_mu, error_sigma, error_rho = C041().c041_initial_uncertainty(type_r, horizon, epsilon, delta, u_select, train_n1_real, train_n1_predict, train_n2_real, train_n2_predict)
        u_l_predict = validation_predict[(index_u_l_predict * horizon):((index_u_l_predict + 1) * horizon)]
        mpc, coefficients, u_data_train, u_data_train_n2, u_data_validation, u_data_test, u_data_train_original = C042().c042_dispatch_model(u_select, error_mu, error_sigma, error_rho, error_bounds, EPS, train_real, train_predict, train_n2_real, train_n2_predict, validation_real, validation_predict, test_real, test_predict, u_l_predict, name_case)

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

        train_real, train_predict, train_n1_real, train_n1_predict, train_n2_real, train_n2_predict, validation_real, validation_predict, test_real, test_predict, error_bounds = C032().c032_calculate_weight(num_groups, weight)
        error_mu, error_sigma, error_rho = C041().c041_initial_uncertainty(b_use_n2, horizon, epsilon, delta, u_select, train_n1_real, train_n1_predict, train_n2_real, train_n2_predict)
        u_l_predict = validation_predict[(index_u_l_predict * horizon):((index_u_l_predict + 1) * horizon)]
        mpc, coefficients, u_data_train, u_data_train_n2, u_data_validation, u_data_test, u_data_train_original = C042().c042_dispatch_model(u_select, error_mu, error_sigma, error_rho, error_bounds, EPS, train_real, train_predict, train_n2_real, train_n2_predict, validation_real, validation_predict, test_real, test_predict, u_l_predict, name_case)

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

        type_r = parameter['type_r']
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

        train_real, train_predict, train_n1_real, train_n1_predict, train_n2_real, train_n2_predict, validation_real, validation_predict, test_real, test_predict, error_bounds = C032().c032_calculate_weight(num_groups, weight)
        error_mu, error_sigma, error_rho = C041().c041_initial_uncertainty(type_r, horizon, epsilon, delta, u_select, train_n1_real, train_n1_predict, train_n2_real, train_n2_predict)
        u_l_predict = validation_predict[(index_u_l_predict * horizon):((index_u_l_predict + 1) * horizon)]
        mpc, coefficients, u_data_train, u_data_train_n2, u_data_validation, u_data_test, u_data_train_original = C042().c042_dispatch_model(u_select, error_mu, error_sigma, error_rho, error_bounds, EPS, train_real, train_predict, train_n2_real, train_n2_predict, validation_real, validation_predict, test_real, test_predict, u_l_predict, name_case)

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

        type_r = parameter['type_r']
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

        train_real, train_predict, train_n1_real, train_n1_predict, train_n2_real, train_n2_predict, validation_real, validation_predict, test_real, test_predict, error_bounds = C032().c032_calculate_weight(num_groups, weight)
        error_mu, error_sigma, error_rho = C041().c041_initial_uncertainty(type_r, horizon, epsilon, delta, u_select, train_n1_real, train_n1_predict, train_n2_real, train_n2_predict)
        u_l_predict = validation_predict[(index_u_l_predict * horizon):((index_u_l_predict + 1) * horizon)]
        mpc, coefficients, u_data_train, u_data_train_n2, u_data_validation, u_data_test, u_data_train_original = C042().c042_dispatch_model(u_select, error_mu, error_sigma, error_rho, error_bounds, EPS, train_real, train_predict, train_n2_real, train_n2_predict, validation_real, validation_predict, test_real, test_predict, u_l_predict, name_case)

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

        type_r = parameter['type_r']
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

        train_real, train_predict, train_n1_real, train_n1_predict, train_n2_real, train_n2_predict, validation_real, validation_predict, test_real, test_predict, error_bounds = C032().c032_calculate_weight(num_groups, weight)
        error_mu, error_sigma, error_rho = C041().c041_initial_uncertainty(type_r, horizon, epsilon, delta, u_select, train_n1_real, train_n1_predict, train_n2_real, train_n2_predict)
        u_l_predict = validation_predict[(index_u_l_predict * horizon):((index_u_l_predict + 1) * horizon)]
        mpc, coefficients, u_data_train, u_data_train_n2, u_data_validation, u_data_test, u_data_train_original = C042().c042_dispatch_model(u_select, error_mu, error_sigma, error_rho, error_bounds, EPS, train_real, train_predict, train_n2_real, train_n2_predict, validation_real, validation_predict, test_real, test_predict, u_l_predict, name_case)

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