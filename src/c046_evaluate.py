## Use validation/test dataset to evaluate the obtained strategy

import numpy as np
import scipy
import gurobipy as gp
from gurobipy import GRB

from utils.case import Case

## Settings
b_validation = False # True: use validation set. False: use test set.
b_reconstruction = True # True: use reconstruction results. False: use the original results.
EPS = 1e-8

## Load case
case = Case()
mpc = case.case_ieee30_modified()
mpc = case.process_case(mpc)

## Load compact form
Areu = scipy.sparse.load_npz('./data/processed/optimization_model/d042_Areu.npz')
Arexc = scipy.sparse.load_npz('./data/processed/optimization_model/d042_Arexc.npz')
Arey = scipy.sparse.load_npz('./data/processed/optimization_model/d042_Arey.npz')
Arixc = scipy.sparse.load_npz('./data/processed/optimization_model/d042_Arixc.npz')
Ariy = scipy.sparse.load_npz('./data/processed/optimization_model/d042_Ariy.npz')
Bre = np.load('./data/processed/optimization_model/d042_Bre.npy')
Bri = np.load('./data/processed/optimization_model/d042_Bri.npy')
Cdxb = np.load('./data/processed/optimization_model/d042_Cdxb.npy')
Cdxc = np.load('./data/processed/optimization_model/d042_Cdxc.npy')
Cry = np.load('./data/processed/optimization_model/d042_Cry.npy')

## Construct uncertainty data from validation/test dataset
if b_validation:
    u_data = np.load('./data/processed/combination/d041_u_data_validation.npy')
else:
    u_data = np.load('./data/processed/combination/d041_u_data_test.npy')

num_data = u_data.shape[0]

# Test whether the points are in the uncertainty set
num_not_in_uncertainty_set = 0
for i in range(num_data):
    if not case.test_u(u_data[i, :], mpc, b_print=True, b_ellipsoid=True):
        print('Data {} is not in the ellipsoid uncertainty set.'.format(i))
        num_not_in_uncertainty_set += 1
print('{} out of {} data are not in the ellipsoid uncertainty set.'.format(num_not_in_uncertainty_set, num_data))

## Load CCG solution
if b_reconstruction:
    sxb = np.load('./data/processed/optimization_output/d045_sxb.npy')
    sxc = np.load('./data/processed/optimization_output/d045_sxc.npy')
    LBUB = np.loadtxt('./data/processed/optimization_output/d045_LBUB.txt')
else:
    sxb = np.load('./data/processed/optimization_output/d043_sxb.npy')
    sxc = np.load('./data/processed/optimization_output/d043_sxc.npy')
    LBUB = np.loadtxt('./data/processed/optimization_output/d043_LBUB.txt')
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
    np.savetxt('./data/processed/optimization_output/d046_cost.txt', cost)

except gp.GurobiError as e:
    print(f"Error code {e.errno}: {e}")

except AttributeError:
    print("Encountered an attribute error")
