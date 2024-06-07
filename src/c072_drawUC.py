import matplotlib.pyplot as plt
import numpy as np

from utils.io import IO

index_u_l_predict = 0
type_u_l = 'test'
num_unit = 6
type_method = 'P2'

for index_unit in range(num_unit):

    x_og, x_pg, x_rp, x_rn, y_rp, y_rn = IO.input_strategy(30, index_u_l_predict, type_u_l, name_method='P2', folder_strategies='./results/strategies/')

    # creating a fake data as example
    time = np.linspace(0, 23, 24)

    # Creating the figure
    plt.figure(figsize=(7,4))

    # Plotting day ahead power output schedule
    plt.plot(time, x_pg[:, index_unit], 'b1-', label='Pre-dispatch')

    # plotting reserve region with fill between function
    plt.fill_between(time, x_pg[:, index_unit] - x_rn[:, index_unit], x_pg[:, index_unit] + x_rp[:, index_unit], 
                    label='Reserve Region', color='gray', alpha=0.4)

    # Plotting re-dispatch power curve
    plt.plot(time, x_pg[:, index_unit] + y_rp[:, index_unit] - y_rn[:, index_unit], 'g2-', label='Worst-case re-dispatch')

    plt.xlabel('Time (h)')
    plt.ylabel('Power Output (100 MW)')
    plt.title('Unit ' + str(index_unit))
    plt.legend()
    plt.grid(True)

plt.show()
