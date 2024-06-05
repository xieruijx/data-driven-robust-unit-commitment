import numpy as np
import gurobipy as gp
from gurobipy import GRB

from utils.combhandler import CombHandler

class C044(object):
    """
    C044 class for uncertainty set reconstruction
    """

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