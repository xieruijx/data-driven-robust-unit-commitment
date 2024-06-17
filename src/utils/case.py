# Data from matpower7.0

import numpy as np

class Case(object):
    """
    Case class for power system cases
    """
    
    @staticmethod
    def case_ieee30():
        """
        IEEE 30-bus case
        """
        mpc = {}

        ## MATPOWER Case Format : Version 2
        mpc['version'] = '2'

        ##-----  Power Flow Data  -----##
        ## system MVA base
        mpc['baseMVA'] = 100

        ## bus data
        #	bus_i	type	Pd	Qd	Gs	Bs	area	Vm	Va	baseKV	zone	Vmax	Vmin
        mpc['bus'] = np.loadtxt('./data/raw/cases/case_ieee30_bus.txt')

        ## generator data
        #	bus	Pg	Qg	Qmax	Qmin	Vg	mBase	status	Pmax	Pmin	Pc1	Pc2	Qc1min	Qc1max	Qc2min	Qc2max	ramp_agc	ramp_10	ramp_30	ramp_q	apf
        mpc['gen'] = np.array([
            [1,	260.2,	-16.1,	10,	0,	1.06,	100,	1,	360.2,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
	        [2,	40,	50,	50,	-40,	1.045,	100,	1,	140,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
	        [5,	0,	37,	40,	-40,	1.01,	100,	1,	100,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
	        [8,	0,	37.3,	40,	-10,	1.01,	100,	1,	100,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
	        [11,	0,	16.2,	24,	-6,	1.082,	100,	1,	100,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
	        [13,	0,	10.6,	24,	-6,	1.071,	100,	1,	100,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0]
        ])

        ## branch data
        #	fbus	tbus	r	x	b	rateA	rateB	rateC	ratio	angle	status	angmin	angmax
        mpc['branch'] = np.loadtxt('./data/raw/cases/case_ieee30_branch.txt')

        ##-----  OPF Data  -----##
        ## generator cost data
        #	1	startup	shutdown	n	x1	y1	...	xn	yn
        #	2	startup	shutdown	n	c(n-1)	...	c0
        mpc['gencost'] = np.array([
            [2,	0,	0,	3,	0.0384319754,	20,	0],
	        [2,	0,	0,	3,	0.25,	20,	0],
	        [2,	0,	0,	3,	0.01,	40,	0],
	        [2,	0,	0,	3,	0.01,	40,	0],
	        [2,	0,	0,	3,	0.01,	40,	0],
	        [2,	0,	0,	3,	0.01,	40,	0]
        ])

        return mpc
    
    @staticmethod
    def case_ieee30_modified(parameter=None):
        """
        Modified IEEE 30-bus case
        """
        mpc = Case().case_ieee30()

        # Modify load
        mpc['bus'][:, 2] = mpc['bus'][:, 2] * 1.1

        mpc['n_t'] = 24 # Number of periods

        #	bus	Pg	Qg	Qmax	Qmin	Vg	mBase	status	Pmax	Pmin	Pc1	Pc2	Qc1min	Qc1max	Qc2min	Qc2max	ramp_agc	ramp_10	ramp_30	ramp_q	apf ramp
        mpc['gen'] = np.array([
            [1,	260.2,	-16.1,	10,	0,	1.06,	100,	1,	360.2,	100,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0, 150],
	        [2,	40,	50,	50,	-40,	1.045,	100,	1,	140,	40,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0, 50],
	        [5,	0,	37,	40,	-40,	1.01,	100,	1,	100,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0, 50],
	        [8,	0,	37.3,	40,	-10,	1.01,	100,	1,	100,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0, 50],
            [11,	0,	16.2,	24,	-6,	1.082,	100,	1,	100,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0, 50],
	        [13,	0,	10.6,	24,	-6,	1.071,	100,	1,	100,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0, 50]
        ])

        mpc['UTDT'] = 6

        #	2	startup	shutdown	n	c(n-1)	...	c0 reserve-up reserve-down ramp-up ramp-down
        mpc['gencost'] = np.array([
            [2,	1,	1,	3,	0,	30,	0, 5, 5, 30, 0],
	        [2,	1,	1,	3,	0,	35,	0, 5, 5, 35, 0],
	        [2,	1,	1,	3,	0,	35,	0, 20, 5, 40, 0],
	        [2,	1,	1,	3,	0,	35,	0, 20, 5, 45, 0],
            [2,	1,	1,	3,	0,	40,	0, 20, 5, 40, 0],
	        [2,	1,	1,	3,	0,	40,	0, 20, 5, 45, 0]
        ])

        #   branch
        mpc['branch'][:, 5] = np.ones(mpc['branch'][:, 5].shape) * mpc['baseMVA']

        #   uncertainty
        bus_uncertain_load = [2, 3, 4, 5, 7, 8, 10,
                           12, 14, 15, 16, 17, 18, 19,
                           20, 21, 23, 24, 26, 29, 30]
        bus_uncertain_wind = [6, 10, 12, 24]
        mpc['bus_uncertain'] = bus_uncertain_load + bus_uncertain_wind
        mpc['u_select'] = parameter['u_select']
        bus_uncertain_load = [b for b, u in zip(bus_uncertain_load, mpc['u_select'][:len(bus_uncertain_load)]) if u]
        bus_uncertain_wind = [b for b, u in zip(bus_uncertain_wind, mpc['u_select'][-len(bus_uncertain_wind):]) if u]
        mpc['bus_uncertain'] = [b for b, u in zip(mpc['bus_uncertain'], mpc['u_select']) if u]
        mpc['n_u'] = len(mpc['bus_uncertain']) # Number of uncertainties

        times_uncertain_load = np.zeros((len(bus_uncertain_load),))
        for i in range(len(bus_uncertain_load)):
            times_uncertain_load[i] = mpc['bus'][bus_uncertain_load[i] - 1, 2] / mpc['baseMVA']
            mpc['bus'][bus_uncertain_load[i] - 1, 2] = 0
        times_uncertain_wind = -50 * np.ones((len(bus_uncertain_wind),)) / mpc['baseMVA']
        mpc['times_uncertain'] = np.concatenate((times_uncertain_load, times_uncertain_wind))

        mpc['u_l_predict'] = parameter['u_l_predict'][:, mpc['u_select']].reshape((-1,))
        mpc['error_mu'] = parameter['error_mu']
        mpc['error_sigma_inv'] = np.linalg.inv(parameter['error_sigma'])
        mpc['error_rho'] = parameter['error_rho']
        error_bounds = parameter['error_bounds']
        error_bounds = error_bounds[mpc['u_select']]
        lb = np.ones((mpc['n_t'], 1)) @ error_bounds[:, 0].reshape((1, -1))
        mpc['error_lb'] = lb.reshape((-1,))
        ub = np.ones((mpc['n_t'], 1)) @ error_bounds[:, 1].reshape((1, -1))
        mpc['error_ub'] = ub.reshape((-1,))

        mpc['u_lu'] = np.ones(mpc['u_l_predict'].shape)
        mpc['u_ll'] = np.zeros(mpc['u_l_predict'].shape)

        return mpc
    
    @staticmethod
    def case_ieee30_parameter(b_faster=False, b_display_SP=False, epsilon=0.05, MaxIter=100, TimeLimit=1):
        """
        Parameters for the IEEE 30-bus unit commitment computation
        """
        parameter = {}
        parameter['b_faster'] = b_faster # False: MILP; True: Mountain climbing for subproblems in CCG
        parameter['b_display_SP'] = b_display_SP
        parameter['num_groups'] = 21
        parameter['num_wind'] = 4
        parameter['horizon'] = 24
        parameter['epsilon'] = epsilon # chance constraint parameter
        parameter['delta'] = 0.05 # probability guarantee parameter
        parameter['MaxIter'] = MaxIter # Maximum iteration number of CCG
        parameter['LargeNumber'] = 1e8 # For the big-M method
        parameter['Tolerance'] = 1e-3 # Tolerance: UB - LB <= Tolerance * UB
        parameter['TimeLimitFC'] = TimeLimit # Time limit of the feasibility check problem
        parameter['TimeLimitSP'] = TimeLimit # Time limit of the subproblem
        parameter['EPS'] = 1e-8 # A small number for margin
        parameter['u_select'] = [False, True, True, False, False, False, True,
                    False, True, True, True, True, True, True,
                    True, False, True, True, True, True, False,
                    True, True, False, False] # Only a part of loads and renewables are uncertain
        return parameter

    @staticmethod
    def case118():
        """
        118-bus case
        """
        mpc = {}

        ## MATPOWER Case Format : Version 2
        mpc['version'] = '2'

        ##-----  Power Flow Data  -----##
        ## system MVA base
        mpc['baseMVA'] = 100

        ## bus data
        #	bus_i	type	Pd	Qd	Gs	Bs	area	Vm	Va	baseKV	zone	Vmax	Vmin
        mpc['bus'] = np.loadtxt('./data/raw/cases/case118_bus.txt')

        ## generator data
        #	bus	Pg	Qg	Qmax	Qmin	Vg	mBase	status	Pmax	Pmin	Pc1	Pc2	Qc1min	Qc1max	Qc2min	Qc2max	ramp_agc	ramp_10	ramp_30	ramp_q	apf
        mpc['gen'] = np.loadtxt('./data/raw/cases/case118_gen.txt')

        ## branch data
        #	fbus	tbus	r	x	b	rateA	rateB	rateC	ratio	angle	status	angmin	angmax
        mpc['branch'] = np.loadtxt('./data/raw/cases/case118_branch.txt')

        ##-----  OPF Data  -----##
        ## generator cost data
        #	1	startup	shutdown	n	x1	y1	...	xn	yn
        #	2	startup	shutdown	n	c(n-1)	...	c0
        mpc['gencost'] = np.loadtxt('./data/raw/cases/case118_gencost.txt')

        return mpc
    
    @staticmethod
    def case118_modified(parameter=None):
        """
        Modified IEEE 118-bus case
        """
        mpc = Case().case118()

        # Modify load
        mpc['bus'][:, 2] = mpc['bus'][:, 2] * 1.0

        mpc['n_t'] = 24 # Number of periods

        #	bus	Pg	Qg	Qmax	Qmin	Vg	mBase	status	Pmax	Pmin	Pc1	Pc2	Qc1min	Qc1max	Qc2min	Qc2max	ramp_agc	ramp_10	ramp_30	ramp_q	apf ramp
        mpc['gen'] = np.concatenate((mpc['gen'], mpc['gen'][:, 8].reshape((-1, 1)) * 0.5), axis=1)

        mpc['UTDT'] = 2

        #	2	startup	shutdown	n	c(n-1)	...	c0 reserve-up reserve-down ramp-up ramp-down
        mpc['gencost'] = np.concatenate((mpc['gencost'], np.full((mpc['gencost'].shape[0], 4), 0)), axis=1)
        mpc['gencost'][:, 5] = mpc['gencost'][:, 4] * mpc['gen'][:, 8] + mpc['gencost'][:, 5]
        mpc['gencost'][:, 4] = 0
        mpc['gencost'][:, 9] = mpc['gencost'][:, 5]
        mpc['gencost'][:, 7] = 5

        #   branch
        mpc['branch'][:, 5] = np.ones(mpc['branch'][:, 5].shape) * mpc['baseMVA']

        #   uncertainty
        bus_uncertain_load = [2, 7, 14, 17, 20, 21, 22,
                           23, 24, 28, 43, 44, 48, 50,
                           51, 52, 57, 58, 72, 73, 84]
        bus_uncertain_wind = [30, 37, 49, 77]
        mpc['bus_uncertain'] = bus_uncertain_load + bus_uncertain_wind
        mpc['u_select'] = parameter['u_select']
        bus_uncertain_load = [b for b, u in zip(bus_uncertain_load, mpc['u_select'][:len(bus_uncertain_load)]) if u]
        bus_uncertain_wind = [b for b, u in zip(bus_uncertain_wind, mpc['u_select'][-len(bus_uncertain_wind):]) if u]
        mpc['bus_uncertain'] = [b for b, u in zip(mpc['bus_uncertain'], mpc['u_select']) if u]
        mpc['n_u'] = len(mpc['bus_uncertain']) # Number of uncertainties

        times_uncertain_load = np.zeros((len(bus_uncertain_load),))
        for i in range(len(bus_uncertain_load)):
            times_uncertain_load[i] = mpc['bus'][bus_uncertain_load[i] - 1, 2] / mpc['baseMVA']
            mpc['bus'][bus_uncertain_load[i] - 1, 2] = 0
        times_uncertain_wind = - 29 * np.ones((len(bus_uncertain_wind),)) / mpc['baseMVA']
        mpc['times_uncertain'] = np.concatenate((times_uncertain_load, times_uncertain_wind))

        mpc['u_l_predict'] = parameter['u_l_predict'][:, mpc['u_select']].reshape((-1,))
        mpc['error_mu'] = parameter['error_mu']
        mpc['error_sigma_inv'] = np.linalg.inv(parameter['error_sigma'])
        mpc['error_rho'] = parameter['error_rho']
        error_bounds = parameter['error_bounds']
        error_bounds = error_bounds[mpc['u_select']]
        lb = np.ones((mpc['n_t'], 1)) @ error_bounds[:, 0].reshape((1, -1))
        mpc['error_lb'] = lb.reshape((-1,))
        ub = np.ones((mpc['n_t'], 1)) @ error_bounds[:, 1].reshape((1, -1))
        mpc['error_ub'] = ub.reshape((-1,))

        mpc['u_lu'] = np.ones(mpc['u_l_predict'].shape)
        mpc['u_ll'] = np.zeros(mpc['u_l_predict'].shape)

        return mpc
    
    @staticmethod
    def case118_parameter(b_faster=True, b_display_SP=False, epsilon=0.05, MaxIter=100, TimeLimit=1):
        """
        Parameters for the IEEE 30-bus unit commitment computation
        """
        parameter = {}
        parameter['b_faster'] = b_faster # False: MILP; True: Mountain climbing for subproblems in CCG
        parameter['b_display_SP'] = b_display_SP
        parameter['num_groups'] = 21
        parameter['num_wind'] = 4
        parameter['horizon'] = 24
        parameter['epsilon'] = epsilon # chance constraint parameter
        parameter['delta'] = 0.05 # probability guarantee parameter
        parameter['MaxIter'] = MaxIter # Maximum iteration number of CCG
        parameter['LargeNumber'] = 1e12 # For the big-M method
        parameter['Tolerance'] = 1e-3 # Tolerance: UB - LB <= Tolerance * UB
        parameter['TimeLimitFC'] = TimeLimit # Time limit of the feasibility check problem
        parameter['TimeLimitSP'] = TimeLimit # Time limit of the subproblem
        parameter['EPS'] = 1e-8 # A small number for margin
        parameter['u_select'] = [True, True, True, True, True, True, True,
                        True, True, True, True, True, True, True,
                        True, True, True, True, True, True, True,
                        True, True, True, True] # Only a part of loads and renewables are uncertain
        return parameter
    
    @staticmethod
    def process_case(mpc):
        """
        Fetch and calculate some parameters
        """
        mpc['branch'] = mpc['branch'][mpc['branch'][:, 10] == 1] # Branches that are on

        mpc['n_b'] = mpc['bus'].shape[0] # Number of buses
        mpc['n_l'] = mpc['branch'].shape[0] # Number of branches
        mpc['n_g'] = mpc['gen'].shape[0] # Number of generators

        mpc['c_x_pg'] = mpc['gencost'][:, 5] * mpc['baseMVA'] # Day-ahead power cost coefficient
        mpc['c_x_rp'] = mpc['gencost'][:, 7] * mpc['baseMVA'] # Up reserve cost coefficient
        mpc['c_x_rn'] = mpc['gencost'][:, 8] * mpc['baseMVA'] # Down reserve cost coefficient
        mpc['c_x_og'] = np.zeros((mpc['n_g'],)) # On cost
        mpc['c_x_ou'] = mpc['gencost'][:, 1] # Turn on cost
        mpc['c_x_od'] = mpc['gencost'][:, 2] # Shut down cost
        mpc['c_y_rp'] = mpc['gencost'][:, 9] * mpc['baseMVA'] # Real-time up cost coefficient
        mpc['c_y_rn'] = mpc['gencost'][:, 10] * mpc['baseMVA'] # Real-time down cost coefficient

        # Bus-branch index matrix
        mpc['bus_branch'] = np.zeros((mpc['n_b'], mpc['n_l'])) # 1: flow out; -1: flow in
        for branch in range(mpc['n_l']):
            mpc['bus_branch'][int(mpc['branch'][branch, 0] - 1), branch] = 1
            mpc['bus_branch'][int(mpc['branch'][branch, 1] - 1), branch] = -1

        # Bus-generator index matrix
        mpc['bus_gen'] = np.zeros((mpc['n_b'], mpc['n_g']))
        for gen in range(mpc['n_g']):
            mpc['bus_gen'][int(mpc['gen'][gen, 0] - 1), gen] = 1

        # Bus-uncertainty index matrix
        mpc['bus_u'] = np.zeros((mpc['n_b'], mpc['n_u']))
        for u in range(mpc['n_u']):
            mpc['bus_u'][mpc['bus_uncertain'][u] - 1, u] = mpc['times_uncertain'][u]
            
        # Daily fixed load
        demand_t = np.array([0.79, 0.73, 0.76, 0.81, 0.90, 0.96,
                             0.95, 0.89, 0.88, 0.83, 0.87, 1.12,
                             1.34, 1.52, 1.56, 1.53, 1.39, 1.02,
                             0.79, 0.71, 0.66, 0.63, 0.60, 0.57]).reshape((-1, 1))
        demand_t = demand_t / demand_t.max()
        mpc['PD'] = demand_t @ mpc['bus'][:, 2].reshape((1, -1)) / mpc['baseMVA'] # Active load power
        mpc['QD'] = demand_t @ mpc['bus'][:, 3].reshape((1, -1)) / mpc['baseMVA'] # Reactive load power

        mpc['R'] = mpc['branch'][:, 2] # Resistance
        mpc['X'] = mpc['branch'][:, 3] # Reactance
        mpc['S'] = mpc['branch'][:, 5] / mpc['baseMVA'] # Capacity

        mpc['Vu'] = mpc['bus'][:, 11] # Upper bound of voltage
        mpc['Vl'] = mpc['bus'][:, 12] # Lower bound of voltage

        mpc['Ramp'] = mpc['gen'][:, 21] / mpc['baseMVA'] # Ramp bound
        mpc['Pmax'] = mpc['gen'][:, 8] / mpc['baseMVA'] # Maximum active generation
        mpc['Pmin'] = mpc['gen'][:, 9] / mpc['baseMVA'] # Minimum active generation
        mpc['Qmax'] = mpc['gen'][:, 3] / mpc['baseMVA'] # Maximum reactive generation
        mpc['Qmin'] = mpc['gen'][:, 4] / mpc['baseMVA'] # Minimum reactive generation

        # Power transfer distribution factor
        mpc['PTDF'] = np.zeros((mpc['n_l'] + mpc['n_b'] - 1, mpc['n_b']))
        Arel = mpc['bus_branch'][1:, :].T
        Mrel1 = np.concatenate((Arel.T, np.zeros((mpc['n_b'] - 1, mpc['n_b'] - 1))), axis=1)
        Mrel2 = np.concatenate((np.diag(mpc['X']), Arel), axis=1)
        Mrel = np.concatenate((Mrel1, Mrel2), axis=0)
        for i in range(1, mpc['n_b']):
            erel = np.zeros((mpc['n_l'] + mpc['n_b'] - 1,))
            erel[i - 1] = 1
            mpc['PTDF'][:, i] = np.linalg.solve(Mrel, erel)
        mpc['PTDF'] = mpc['PTDF'][:mpc['n_l'], :]
        mpc['PTDF'][np.abs(mpc['PTDF']) < 1e-8] = 0

        return mpc
    