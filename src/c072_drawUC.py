import matplotlib.pyplot as plt
import numpy as np

from utils.io import IO

index_u_l_predict = 0
type_u_l = 'test'
num_unit = 6
type_method = 'P2'

fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(9, 5))

for index_unit, ax in enumerate(axs.flatten()):

    x_og, x_pg, x_rp, x_rn, y_rp, y_rn = IO.read_strategy(30, index_u_l_predict, type_u_l, name_method='P2', folder_strategies='./results/strategies/')
    x_pg = x_pg * 100
    x_rp = x_rp * 100
    x_rn = x_rn * 100
    y_rp = y_rp * 100
    y_rn = y_rn * 100

    time = np.linspace(0, 23, 24)

    ax.plot(time, x_pg[:, index_unit], 'b*-', label='Pre-dispatch')
    ax.fill_between(time, x_pg[:, index_unit] - x_rn[:, index_unit], x_pg[:, index_unit] + x_rp[:, index_unit], 
                    label='Reserve', color='gray', alpha=0.4)
    ax.plot(time, x_pg[:, index_unit] + y_rp[:, index_unit] - y_rn[:, index_unit], 'g2-', label='Re-dispatch')

    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Power of unit ' + str(index_unit + 1) + ' (MW)')
    ax.legend()
    ax.grid(linestyle='--')

plt.show()
