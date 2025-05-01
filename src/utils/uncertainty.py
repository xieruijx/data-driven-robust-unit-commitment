import numpy as np
import gurobipy as gp
from gurobipy import GRB

class Uncertainty(object):
    """
    Uncertainty class for uncertainty testing and revision
    """
    
    @staticmethod
    def test_u(su, mpc, b_print=False, b_ellipsoid=True):
        """
        Test whether u is in the ellipsoid uncertainty set
        """
        # bounds of uncertainty
        if not np.all(su >= mpc['u_ll']):
            if b_print:
                print('!!!!!!!!!!!Lower bound of uncertainty does not hold!!!!!!!!!!!')
                print('The largest violation is: {}'.format(np.max(mpc['u_ll'] - su)))
            return False
        if not np.all(su <= mpc['u_lu']):
            if b_print:
                print('!!!!!!!!!!!Upper bound of uncertainty does not hold!!!!!!!!!!!')
                print('The largest violation is: {}'.format(np.max(su - mpc['u_lu'])))
            return False
        # bounds of error
        error = mpc['u_l_predict'] - su
        if not np.all(error >= mpc['error_lb']):
            if b_print:
                print('!!!!!!!!!!!!!Lower bound of error does not hold!!!!!!!!!!!!!!')
                print('The largest violation is: {}'.format(np.max(mpc['error_lb'] - error)))
            return False
        if not np.all(error <= mpc['error_ub']):
            if b_print:
                print('!!!!!!!!!!!!!Upper bound of error does not hold!!!!!!!!!!!!!!')
                print('The largest violation is: {}'.format(np.max(error - mpc['error_ub'])))
            return False
        # quadratic
        if b_ellipsoid:
            u_ld = error - mpc['error_mu'] # error - mu
            if u_ld @ mpc['error_sigma_inv'] @ u_ld > mpc['error_rho']:
                if b_print:
                    print('!!!!!!!!!!!!!Quadratic constraint does not hold!!!!!!!!!!!!!!')
                    print('The violation is: {}'.format(u_ld @ mpc['error_sigma_inv'] @ u_ld - mpc['error_rho']))
                return False

        return True
    
    @staticmethod
    def revise_u(su, mpc, EPS=1e-8, b_print=False, b_ellipsoid=True):
        """
        Revise a solution that is not in the ellipsoid uncertainty set to be in the uncertainty set
        """
        print('Revise uncertainty.')

        # quadratic
        if b_ellipsoid:
            u_ld = mpc['u_l_predict'] - su - mpc['error_mu']
            quadratic = (1 + EPS) * u_ld @ mpc['error_sigma_inv'] @ u_ld
            if quadratic > mpc['error_rho']:
                u_ld = u_ld / np.sqrt(quadratic / mpc['error_rho'])
                su = mpc['u_l_predict'] - u_ld - mpc['error_mu']
        # bounds of error
        error = mpc['u_l_predict'] - su
        error = np.minimum(error, mpc['error_ub'] - EPS * EPS)
        error = np.maximum(error, mpc['error_lb'] + EPS * EPS)
        su = mpc['u_l_predict'] - error
        # bounds of uncertainty
        su = np.minimum(su, mpc['u_lu'] - EPS * EPS)
        su = np.maximum(su, mpc['u_ll'] + EPS * EPS)

        # Test the first time. If not feasible, use optimization to find a projection
        if b_print:
            print('test_u in revise_u')
        if not Uncertainty().test_u(su, mpc, b_print=b_print, b_ellipsoid=b_ellipsoid):
            m = gp.Model('m')
            u = m.addMVar(su.shape, lb=-float('inf'), vtype=GRB.CONTINUOUS)
            u_ld = m.addMVar(su.shape, lb=-float('inf'), vtype=GRB.CONTINUOUS) # error - mu
            m.addConstr(u_ld == mpc['u_l_predict'] - u - mpc['error_mu'])
            m.addConstr(u >= mpc['u_ll'] + EPS)
            m.addConstr(mpc['u_lu'] >= u + EPS)
            m.addConstr(mpc['u_l_predict'] - u >= mpc['error_lb'] + EPS)
            m.addConstr(mpc['error_ub'] >= mpc['u_l_predict'] - u + EPS)
            m.addMQConstr(Q=mpc['error_sigma_inv'], c=None, sense='<', rhs=mpc['error_rho'] / (1 + EPS), xQ_L=u_ld, xQ_R=u_ld)
            m.setObjective(gp.quicksum((u - su) * (u - su)), GRB.MINIMIZE)
            m.setParam('OutputFlag', 0) 
            m.optimize()
            su = u.X
            # Test the second time
            if not Uncertainty().test_u(su, mpc, b_print=b_print, b_ellipsoid=b_ellipsoid):
                raise RuntimeError('Failure of uncertainty revision')

        return su
    