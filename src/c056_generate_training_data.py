## Provide training data for the surrogate model of optimizing weight
import numpy as np

from utils.optimization import Optimization
from utils.case import Case
from utils.io import IO

## Settings
type_u_l = 'train'
number_u_l = 336
file_numbers = './data/processed/weight/d056_number.txt'
num_sample = 10 # the number of sample for each group of data
parameter = Case().case_ieee30_parameter()
low = 0
high = 1

# ## Initiation
# numbers = [0 for _ in range(number_u_l)]
# with open(file_numbers, 'w') as f:
#     f.write('\n'.join(map(str, numbers)))

## Read the current status
with open(file_numbers, 'r') as f:
    numbers = [int(line.strip()) for line in f]

while np.min(np.array(numbers)) < num_sample:
    # Find the first group of data that is not enough
    for index_u_l_predict, number in enumerate(numbers):
        if number < num_sample:
            print('This is ' + type_u_l + str(index_u_l_predict) + ' sample' + str(number))
            # Set weight as a random value
            while True:
                r1 = np.random.uniform(low=low, high=high)
                r2 = np.random.uniform(low=low, high=high)
                r3 = 1 - r1 - r2
                if (r3 >= low) & (r3 <= high):
                    break
            weight = np.array([r1, r2, r3])
            print('The random weight is: {}'.format(weight))

            # Optimization and output
            validation_cost, test_cost, sxb1, sxc1, LBUB1, sxb2, sxc2, LBUB2, time, train_cost, train_order, interpret = Optimization().weight2cost(parameter, weight, 'n1', None, index_u_l_predict, 'case_ieee30', type_u_l)
            IO().output_UC(index_u_l_predict, type_u_l, weight, train_cost, train_order, validation_cost, test_cost, LBUB1, LBUB2, time, interpret, 'Proposed', folder_outputs='./data/processed/weight/outputs/n' + str(number) + '_', folder_strategies='./data/processed/weight/strategies/n' + str(number) + '_')

            # Update the current status
            numbers[index_u_l_predict] = number + 1
            with open(file_numbers, 'w') as f:
                f.write('\n'.join(map(str, numbers)))

            break