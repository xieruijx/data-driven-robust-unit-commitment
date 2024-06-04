## CCG with the reconstructed uncertainty set

import numpy as np
import scipy
import time
import gurobipy as gp
from gurobipy import GRB

from utils.case import Case

## Time
time_start = time.time()

## Settings
LargeNumber = 1e8 # For the big-M method
Tolerance = 1e-3 # Tolerance: UB - LB <= Tolerance * UB
TimeLimitFC = 10 # Time limit of the feasibility check problem
TimeLimitSP = 100 # Time limit of the subproblem
MaxIter = 100 # Maximum iteration number of CCG
EPS = 1e-8 # A small number for margin

## Load case
case = Case()
mpc = case.case_ieee30_modified()
mpc = case.process_case(mpc)

## Load compact form
Adexb = scipy.sparse.load_npz('./data/processed/optimization_model/d042_Adexb.npz')
Adexc = scipy.sparse.load_npz('./data/processed/optimization_model/d042_Adexc.npz')
Adixb = scipy.sparse.load_npz('./data/processed/optimization_model/d042_Adixb.npz')
Adixc = scipy.sparse.load_npz('./data/processed/optimization_model/d042_Adixc.npz')
Areu = scipy.sparse.load_npz('./data/processed/optimization_model/d042_Areu.npz')
Arexc = scipy.sparse.load_npz('./data/processed/optimization_model/d042_Arexc.npz')
Arey = scipy.sparse.load_npz('./data/processed/optimization_model/d042_Arey.npz')
Arixc = scipy.sparse.load_npz('./data/processed/optimization_model/d042_Arixc.npz')
Ariy = scipy.sparse.load_npz('./data/processed/optimization_model/d042_Ariy.npz')
Aueu = scipy.sparse.load_npz('./data/processed/optimization_model/d044_Aueu.npz')
Auey = scipy.sparse.load_npz('./data/processed/optimization_model/d044_Auey.npz')
Auiy = scipy.sparse.load_npz('./data/processed/optimization_model/d044_Auiy.npz')
Bde = np.load('./data/processed/optimization_model/d042_Bde.npy')
Bdi = np.load('./data/processed/optimization_model/d042_Bdi.npy')
Bre = np.load('./data/processed/optimization_model/d042_Bre.npy')
Bri = np.load('./data/processed/optimization_model/d042_Bri.npy')
Bue = np.load('./data/processed/optimization_model/d044_Bue.npy')
Bui = np.load('./data/processed/optimization_model/d044_Bui.npy')
Cdxb = np.load('./data/processed/optimization_model/d042_Cdxb.npy')
Cdxc = np.load('./data/processed/optimization_model/d042_Cdxc.npy')
Cry = np.load('./data/processed/optimization_model/d042_Cry.npy')

## Load uncertainty data
u_data = np.load('./data/processed/uncertainty/d044_u_data.npy')
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
    np.save('./data/processed/optimization_output/d045_u_data.npy', u_data)
    np.save('./data/processed/optimization_output/d045_ulist.npy', ulist)
    np.save('./data/processed/optimization_output/d045_sxb.npy', sxb)
    np.save('./data/processed/optimization_output/d045_sxc.npy', sxc)
    LBUB = np.concatenate((LB[:(Iter + 1)].reshape((-1, 1)), UBU[:(Iter + 1)].reshape((-1, 1)), UBL[:(Iter + 1)].reshape((-1, 1))), axis=1)
    np.savetxt('./data/processed/optimization_output/d045_LBUB.txt', LBUB)

except gp.GurobiError as e:
    print(f"Error code {e.errno}: {e}")

except AttributeError:
    print("Encountered an attribute error")

time_elapsed = time.time() - time_start
with open('./data/processed/optimization_output/d045_time.txt', 'w') as f:
    f.write(str(time_elapsed))
