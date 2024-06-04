## Uncertainty set reconstruction

import numpy as np
import scipy
import gurobipy as gp
from gurobipy import GRB

from utils.case import Case
from utils.combhandler import CombHandler

## Settings
EPS = 1e-8
epsilon = 0.05 # chance constraint parameter
delta = 0.05 # probability guarantee parameter

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

## Load u_data_train_n2
u_data = np.load('./data/processed/combination/d041_u_data_train_n2.npy')
num_data = u_data.shape[0]

# Test whether the points are in the ellipsoid uncertainty set
for i in range(num_data):
    if not case.test_u(u_data[i, :], mpc, b_print=True, b_ellipsoid=True):
        print('Data {} is not in the uncertainty set.'.format(i))

## Load CCG solution
sxb = np.load('./data/processed/optimization_output/d043_sxb.npy')
sxc = np.load('./data/processed/optimization_output/d043_sxc.npy')
LBUB = np.loadtxt('./data/processed/optimization_output/d043_LBUB.txt')
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
    with open('./data/processed/uncertainty/d044_radius.txt', 'w') as f:
        f.write(str(radius))
    u_data = u_data[np.argsort(sos)[:rank]]
    np.save('./data/processed/uncertainty/d044_u_data.npy', u_data)

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
    scipy.sparse.save_npz('./data/processed/optimization_model/d044_Aueu.npz', Aueu)
    Auey = A[:numue, numuu:]
    scipy.sparse.save_npz('./data/processed/optimization_model/d044_Auey.npz', Auey)
    Bue = B[:numue]
    np.save('./data/processed/optimization_model/d044_Bue.npy', Bue)
    Auiy = A[numue:, numuu:]
    scipy.sparse.save_npz('./data/processed/optimization_model/d044_Auiy.npz', Auiy)
    Bui = B[numue:]
    np.save('./data/processed/optimization_model/d044_Bui.npy', Bui)

    ## Test the output model
    tmodel = gp.Model('Test')
    tu = tmodel.addMVar((Aueu.shape[1],), lb=-float('inf'), vtype=GRB.CONTINUOUS)
    ty = tmodel.addMVar((Auey.shape[1],), lb=-float('inf'), vtype=GRB.CONTINUOUS)
    tmodel.addConstr(Aueu @ tu + Auey @ ty == Bue, 'ue')
    tmodel.addConstr(Auiy @ ty >= Bui, 'ui')
    tmodel.setObjective(0, GRB.MINIMIZE)
    tmodel.optimize()

except gp.GurobiError as e:
    print(f"Error code {e.errno}: {e}")

except AttributeError:
    print("Encountered an attribute error")
