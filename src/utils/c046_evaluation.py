import numpy as np
import gurobipy as gp
from gurobipy import GRB

class C046(object):
    """
    C046 class for evaluation
    """
    
    @staticmethod
    def c046_evaluate_faster(u_data, coefficients, sxb, sxc, LBUB, b_sort=True):
        """
        Use dataset to evaluate the obtained strategy
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

            if b_sort:
                cost = np.sort(cost)[::-1]

        return cost