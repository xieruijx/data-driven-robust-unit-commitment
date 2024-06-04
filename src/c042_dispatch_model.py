## Generate dispatch model and the coefficients of the optimization problem

import numpy as np
import scipy
import gurobipy as gp
from gurobipy import GRB

from utils.case import Case

## Load case
mpc = Case().case_ieee30_modified()
mpc = Case().process_case(mpc)
u_l_predict = mpc['u_l_predict'].reshape((mpc['n_t'], mpc['n_u']))

## Define variables and get model
try:
    # Create a new model
    model = gp.Model('Modeling')

    # Create variables
    num_var = np.zeros((4,)) # Uncertain, day-ahead binary, day-ahead continuous, real-time

    u_l = model.addMVar((mpc['n_t'], mpc['n_u']), lb=-float('inf'), vtype=GRB.CONTINUOUS) # Uncertain load demand
    model.update()
    num_var[0] = model.NumVars - 0
    
    x_og = model.addMVar((mpc['n_t'], mpc['n_g']), vtype=GRB.BINARY) # Day-ahead generator on
    x_ou = model.addMVar((mpc['n_t'], mpc['n_g']), vtype=GRB.BINARY) # Day-ahead generator up
    x_od = model.addMVar((mpc['n_t'], mpc['n_g']), vtype=GRB.BINARY) # Day-ahead generator down
    model.update()
    num_var[1] = model.NumVars - 0

    x_pg = model.addMVar((mpc['n_t'], mpc['n_g']), lb=-float('inf'), vtype=GRB.CONTINUOUS) # Day-ahead active power generation
    x_rp = model.addMVar((mpc['n_t'], mpc['n_g']), lb=-float('inf'), vtype=GRB.CONTINUOUS) # Day-ahead upper power reserve
    x_rn = model.addMVar((mpc['n_t'], mpc['n_g']), lb=-float('inf'), vtype=GRB.CONTINUOUS) # Day-ahead down power reserve
    model.update()
    num_var[2] = model.NumVars - 0
    
    y_rp = model.addMVar((mpc['n_t'], mpc['n_g']), lb=-float('inf'), vtype=GRB.CONTINUOUS) # Real-time upper power
    y_rn = model.addMVar((mpc['n_t'], mpc['n_g']), lb=-float('inf'), vtype=GRB.CONTINUOUS) # Real-time down power
    model.update()
    num_var[3] = model.NumVars - 0

    # Set objective
    o_x_pg = gp.quicksum(mpc['c_x_pg'][g] * x_pg[t, g] for t in range(mpc['n_t']) for g in range(mpc['n_g']))
    o_x_rp = gp.quicksum(mpc['c_x_rp'][g] * x_rp[t, g] for t in range(mpc['n_t']) for g in range(mpc['n_g']))
    o_x_rn = gp.quicksum(mpc['c_x_rn'][g] * x_rn[t, g] for t in range(mpc['n_t']) for g in range(mpc['n_g']))
    o_x_og = gp.quicksum(mpc['c_x_og'][g] * x_og[t, g] for t in range(mpc['n_t']) for g in range(mpc['n_g']))
    o_x_ou = gp.quicksum(mpc['c_x_ou'][g] * x_ou[t, g] for t in range(mpc['n_t']) for g in range(mpc['n_g']))
    o_x_od = gp.quicksum(mpc['c_x_od'][g] * x_od[t, g] for t in range(mpc['n_t']) for g in range(mpc['n_g']))
    o_x = gp.quicksum([o_x_pg, o_x_rp, o_x_rn, o_x_og, o_x_ou, o_x_od])
    o_y_rp = gp.quicksum(mpc['c_y_rp'][g] * y_rp[t, g] for t in range(mpc['n_t']) for g in range(mpc['n_g']))
    o_y_rn = gp.quicksum(mpc['c_y_rn'][g] * y_rn[t, g] for t in range(mpc['n_t']) for g in range(mpc['n_g']))
    o_y = gp.quicksum([o_y_rp, o_y_rn])
    model.setObjective(gp.quicksum([o_x, o_y]), GRB.MINIMIZE)

    # Add constraints
    num_con = np.zeros((4,)) # Day-ahead =, day-ahead >, real-time =, real-time >

    # Day-ahead
    # Power flow 1
    model.addConstrs((x_pg[t, :] @ np.ones((mpc['n_g'],)) == mpc['PD'][t, :] @ np.ones((mpc['n_b'],)) + u_l_predict[t, :] @ mpc['bus_u'].T @ np.ones((mpc['n_b'],)) for t in range(mpc['n_t'])), name='d_pf_e')
    # Unit commitment 1
    model.addConstr(x_og[1:, :] - x_og[:-1, :] == x_ou[1:, :] - x_od[1:, :], name='d_UC_logical')
    model.update()
    num_con[0] = model.NumConstrs - 0

    # Power flow 2
    model.addConstrs((mpc['PTDF'][l, :] @ (x_pg[t, :] @ mpc['bus_gen'].T - mpc['PD'][t, :] - u_l_predict[t, :] @ mpc['bus_u'].T) >= - mpc['S'][l] for t in range(mpc['n_t']) for l in range(mpc['n_l'])), name='d_pf_l_lb')
    model.addConstrs((mpc['S'][l] >= mpc['PTDF'][l, :] @ (x_pg[t, :] @ mpc['bus_gen'].T - mpc['PD'][t, :] - u_l_predict[t, :] @ mpc['bus_u'].T) for t in range(mpc['n_t']) for l in range(mpc['n_l'])), name='d_pf_l_ub')
    # Generator dispatch
    model.addConstrs((x_rp[t, :] >= 0 for t in range(mpc['n_t'])), name='d_ED_rp_lb')
    model.addConstrs((mpc['Ramp'] * x_og[t, :] >= x_rp[t, :] for t in range(mpc['n_t'])), name='d_ED_rp_ub')
    model.addConstrs((x_rn[t, :] >= 0 for t in range(mpc['n_t'])), name='d_ED_rn_lb')
    model.addConstrs((mpc['Ramp'] * x_og[t, :] >= x_rn[t, :] for t in range(mpc['n_t'])), name='d_ED_rn_ub')
    model.addConstrs((x_pg[t, :] >= mpc['Pmin'] * x_og[t, :] + x_rn[t, :] for t in range(mpc['n_t'])), name='d_ED_pg_lb')
    model.addConstrs((mpc['Pmax'] * x_og[t, :] - x_rp[t, :] >= x_pg[t, :] for t in range(mpc['n_t'])), name='d_ED_pg_ub')
    model.addConstrs((mpc['Ramp'] * x_og[t, :] + mpc['Pmax'] * x_ou[t + 1, :] >= x_pg[t + 1, :] + x_rp[t + 1, :] - x_pg[t, :] + x_rn[t, :] for t in range(mpc['n_t'] - 1)), name='d_ED_ramp_u')
    model.addConstrs((x_pg[t + 1, :] - x_rn[t + 1, :] - x_pg[t, :] - x_rp[t, :] >= - mpc['Ramp'] * x_og[t + 1, :] - mpc['Pmax'] * x_od[t + 1, :] for t in range(mpc['n_t'] - 1)), name='d_ED_ramp_d')
    # Unit commitment 2
    model.addConstrs((gp.quicksum(x_og[t:(t + mpc['UTDT']), g]) >= mpc['UTDT'] * x_ou[t, g] for t in range(mpc['n_t'] - mpc['UTDT'] + 1) for g in range(mpc['n_g'])), name='d_UC_ou1')
    model.addConstrs((gp.quicksum(x_og[t:, g]) >= (mpc['n_t'] - t) * x_ou[t, g] for t in range(mpc['n_t'] - mpc['UTDT'] + 1, mpc['n_t']) for g in range(mpc['n_g'])), name='d_UC_ou2')
    model.addConstrs((mpc['UTDT'] - gp.quicksum(x_og[t:(t + mpc['UTDT']), g]) >= mpc['UTDT'] * x_od[t, g] for t in range(mpc['n_t'] - mpc['UTDT'] + 1) for g in range(mpc['n_g'])), name='d_UC_od1')
    model.addConstrs((mpc['n_t'] - t - gp.quicksum(x_og[t:, g]) >= (mpc['n_t'] - t) * x_od[t, g] for t in range(mpc['n_t'] - mpc['UTDT'] + 1, mpc['n_t']) for g in range(mpc['n_g'])), name='d_UC_od2')
    model.addConstrs((1 >= x_ou[t, g] + x_od[t, g] for t in range(mpc['n_t']) for g in range(mpc['n_g'])), name='d_UC_complementary')
    model.update()
    num_con[1] = model.NumConstrs - 0

    # Real-time
    # Power flow
    model.addConstrs(((x_pg[t, :] + y_rp[t, :] - y_rn[t, :]) @ np.ones((mpc['n_g'],)) == mpc['PD'][t, :] @ np.ones((mpc['n_b'],)) + u_l[t, :] @ mpc['bus_u'].T @ np.ones((mpc['n_b'],)) for t in range(mpc['n_t'])), name='r_pf_e') # the only slack
    model.update()
    num_con[2] = model.NumConstrs - 0

    model.addConstrs((mpc['PTDF'][l, :] @ ((x_pg[t, :] + y_rp[t, :] - y_rn[t, :]) @ mpc['bus_gen'].T - mpc['PD'][t, :] - u_l[t, :] @ mpc['bus_u'].T) >= - mpc['S'][l] for t in range(mpc['n_t']) for l in range(mpc['n_l'])), name='r_pf_l_lb')
    model.addConstrs((mpc['S'][l] >= mpc['PTDF'][l, :] @ ((x_pg[t, :] + y_rp[t, :] - y_rn[t, :]) @ mpc['bus_gen'].T - mpc['PD'][t, :] - u_l[t, :] @ mpc['bus_u'].T) for t in range(mpc['n_t']) for l in range(mpc['n_l'])), name='r_pf_l_ub')
    # Generator dispatch
    model.addConstrs((x_rp[t, :] >= y_rp[t, :] for t in range(mpc['n_t'])), name='r_ED_rp_ub')
    model.addConstrs((x_rn[t, :] >= y_rn[t, :] for t in range(mpc['n_t'])), name='r_ED_rn_ub')
    model.addConstrs((y_rp[t, :] >= 0 for t in range(mpc['n_t'])), name='r_ED_rp_lb')
    model.addConstrs((y_rn[t, :] >= 0 for t in range(mpc['n_t'])), name='r_ED_rn_lb')
    model.update()
    num_con[3] = model.NumConstrs - 0

    ## Output model coefficients
    A = model.getA()
    B = np.array(model.getAttr('RHS', model.getConstrs()))
    C = np.array(model.getAttr('Obj',model.getVars()))
    sense = model.getAttr('Sense', model.getConstrs())
    for i, x in enumerate(sense):
        if x == '<':
            A[i, :] = - A[i, :]
            B[i] = - B[i]
    
    # Day-ahead equation: Adexb * xb + Adexc * xc = Bde
    Adexb = A[:int(num_con[0]), int(num_var[0]):int(num_var[1])]
    scipy.sparse.save_npz('./data/processed/optimization_model/d042_Adexb.npz', Adexb)
    Adexc = A[:int(num_con[0]), int(num_var[1]):int(num_var[2])]
    scipy.sparse.save_npz('./data/processed/optimization_model/d042_Adexc.npz', Adexc)

    # Day-ahead inequality: Adixb * xb + Adixc * xc >= Bdi
    Adixb = A[int(num_con[0]):int(num_con[1]), int(num_var[0]):int(num_var[1])]
    scipy.sparse.save_npz('./data/processed/optimization_model/d042_Adixb.npz', Adixb)
    Adixc = A[int(num_con[0]):int(num_con[1]), int(num_var[1]):int(num_var[2])]
    scipy.sparse.save_npz('./data/processed/optimization_model/d042_Adixc.npz', Adixc)

    # Real-time equation: Areu * u + Arexc * xc + Arey * y = Bre
    Areu = A[int(num_con[1]):int(num_con[2]), :int(num_var[0])]
    scipy.sparse.save_npz('./data/processed/optimization_model/d042_Areu.npz', Areu)
    Arexc = A[int(num_con[1]):int(num_con[2]), int(num_var[1]):int(num_var[2])]
    scipy.sparse.save_npz('./data/processed/optimization_model/d042_Arexc.npz', Arexc)
    Arey = A[int(num_con[1]):int(num_con[2]), int(num_var[2]):int(num_var[3])]
    scipy.sparse.save_npz('./data/processed/optimization_model/d042_Arey.npz', Arey)

    # Real-time inequality: Arixc * xc + Ariy * y >= Bri
    Arixc = A[int(num_con[2]):int(num_con[3]), int(num_var[1]):int(num_var[2])]
    scipy.sparse.save_npz('./data/processed/optimization_model/d042_Arixc.npz', Arixc)
    Ariy = A[int(num_con[2]):int(num_con[3]), int(num_var[2]):int(num_var[3])]
    scipy.sparse.save_npz('./data/processed/optimization_model/d042_Ariy.npz', Ariy)

    # Right-hand side
    Bde = B[:int(num_con[0])]
    np.save('./data/processed/optimization_model/d042_Bde.npy', Bde)
    Bdi = B[int(num_con[0]):int(num_con[1])]
    np.save('./data/processed/optimization_model/d042_Bdi.npy', Bdi)
    Bre = B[int(num_con[1]):int(num_con[2])]
    np.save('./data/processed/optimization_model/d042_Bre.npy', Bre)
    Bri = B[int(num_con[2]):int(num_con[3])]
    np.save('./data/processed/optimization_model/d042_Bri.npy', Bri)

    # Objective
    Cdxb = C[int(num_var[0]):int(num_var[1])]
    np.save('./data/processed/optimization_model/d042_Cdxb.npy', Cdxb)
    Cdxc = C[int(num_var[1]):int(num_var[2])]
    np.save('./data/processed/optimization_model/d042_Cdxc.npy', Cdxc)
    Cry = C[int(num_var[2]):int(num_var[3])]
    np.save('./data/processed/optimization_model/d042_Cry.npy', Cry)

    ## Test the compact problem
    tmodel = gp.Model('Test')
    u = tmodel.addMVar((int(num_var[0]),), lb=-float('inf'), vtype=GRB.CONTINUOUS)
    xb = tmodel.addMVar((int(num_var[1] - num_var[0]),), vtype=GRB.BINARY)
    xc = tmodel.addMVar((int(num_var[2] - num_var[1]),), lb=-float('inf'), vtype=GRB.CONTINUOUS)
    y = tmodel.addMVar((int(num_var[3] - num_var[2]),), lb=-float('inf'), vtype=GRB.CONTINUOUS)
    tmodel.setObjective(Cdxb @ xb + Cdxc @ xc + Cry @ y, GRB.MINIMIZE)
    tmodel.addConstr(Adexb @ xb + Adexc @ xc == Bde, name='de')
    tmodel.addConstr(Adixb @ xb + Adixc @ xc >= Bdi, name='di')
    tmodel.addConstr(Areu @ u + Arexc @ xc + Arey @ y == Bre, name='re')
    tmodel.addConstr(Arixc @ xc + Ariy @ y >= Bri, name='ri')
    tmodel.optimize()
    
except gp.GurobiError as e:
    print(f"Error code {e.errno}: {e}")

except AttributeError:
    print("Encountered an attribute error")
    