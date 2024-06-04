## Generate random weight and calculate the cost

import numpy as np
import pandas as pd

from utils.optimization import Optimization

optimization = Optimization()

## Settings
name_case = 'case118'

parameter = {}
parameter['b_use_n2'] = False # False: the radius is the largest one in n1. True: the radius is the rank one in n2.
parameter['b_display_SP'] = False
parameter['num_groups'] = 21
parameter['horizon'] = 24
parameter['epsilon'] = 0.05 # chance constraint parameter
parameter['delta'] = 0.05 # probability guarantee parameter
parameter['MaxIter'] = 100 # Maximum iteration number of CCG
parameter['LargeNumber'] = 1e12 # For an initial upper bound
parameter['Tolerance'] = 1e-3 # Tolerance: UB - LB <= Tolerance * UB
parameter['TimeLimitFC'] = 10 # Time limit of the feasibility check problem
parameter['TimeLimitSP'] = 10 # Time limit of the subproblem
parameter['EPS'] = 1e-8 # A small number for margin
parameter['u_select'] = [True, True, True, True, True, True, True,
                        True, True, True, True, True, True, True,
                        True, True, True, True, True, True, True,
                        True, True] # Only a part of loads and renewables are uncertain

index_u_l_predict = 0

low = -0.2
high = 1.4

while True:

    numbers = pd.read_csv('./data/processed/weight118/number.txt', header=None)
    numbers = numbers[0]
    number = numbers.iloc[-1] + 1
    with open('./data/processed/weight118/number.txt', 'w') as f:
        f.write(str(number) + '\n')

    ## Set weight as a random value
    while True:
        r1 = np.random.uniform(low=low, high=high)
        r2 = np.random.uniform(low=low, high=high)
        r3 = 1 - r1 - r2
        if (r3 >= low) & (r3 <= high):
            break
    weight = np.array([r1, r2, r3])
    print('The random weight is: {}'.format(weight))
    np.savetxt('./data/processed/weight118/index_' + str(index_u_l_predict) + '_weight_' + str(number) + '.txt', weight)

    ## Processing
    validation_cost, test_cost, sxb1, sxc1, LBUB1, time1, sxb2, sxc2, LBUB2, time2 = optimization.weight2cost(parameter, weight, index_u_l_predict, name_case)
    cost = np.concatenate((validation_cost.reshape((-1, 1)), test_cost.reshape((-1, 1))), axis=1)
    
    np.savetxt('./data/processed/weight118/index_' + str(index_u_l_predict) + '_cost_' + str(number) + '.txt', cost)
    np.savetxt('./data/processed/weight118/index_' + str(index_u_l_predict) + '_LBUB1_' + str(number) + '.txt', LBUB1)
    np.savetxt('./data/processed/weight118/index_' + str(index_u_l_predict) + '_LBUB2_' + str(number) + '.txt', LBUB2)
