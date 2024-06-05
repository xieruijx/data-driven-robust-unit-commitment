import numpy as np
import gurobipy as gp
from gurobipy import GRB

class C046(object):
    """
    C046 class for evaluation
    """

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