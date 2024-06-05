import numpy as np
import time
import gurobipy as gp
from gurobipy import GRB

from utils.case import Case

class C047(object):
    """
    C047 class for optimization with respect to a discrete uncertainty set
    """

    @staticmethod
    def c047_approx(rank, MaxIter, coefficients, u_data_train_original):
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
    def c047_MILP(epsilon, LargeNumber, MaxIter, coefficients, u_data_train_original):
        """
        Using a portion of data and MILP
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
    