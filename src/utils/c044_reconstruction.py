import numpy as np
import gurobipy as gp
from gurobipy import GRB

from utils.combhandler import CombHandler
from utils.c046_evaluation import C046

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

        cost = C046().c046_evaluate_faster(u_data, coefficients, sxb, sxc, LBUB)
        print('There are {} data in n2, where {} of them are not feasible.'.format(num_data, np.count_nonzero(cost == float('inf'))))

        # Get radius and feasible uncertainty data
        rank = CombHandler().get_rank(num_data, epsilon, delta)
        radius = cost[np.argsort(cost)[rank - 1]]
        u_data = u_data[np.argsort(cost)[:rank]]

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
            umodel.addConstr(radius >= Cdxb @ sxb + Cdxc @ sxc + Cry @ uy, name='obj')
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
    