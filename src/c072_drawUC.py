import matplotlib.pyplot as plt
import numpy as np

index_u_l_predict = 9
index_unit = 0

x_og = np.loadtxt('./data/processed/weight/index_' + str(index_u_l_predict) + '_x_og_Proposed.txt')
x_pg = np.loadtxt('./data/processed/weight/index_' + str(index_u_l_predict) + '_x_pg_Proposed.txt')
x_rp = np.loadtxt('./data/processed/weight/index_' + str(index_u_l_predict) + '_x_rp_Proposed.txt')
x_rn = np.loadtxt('./data/processed/weight/index_' + str(index_u_l_predict) + '_x_rn_Proposed.txt')
y_rp = np.loadtxt('./data/processed/weight/index_' + str(index_u_l_predict) + '_y_rp_Proposed.txt')
y_rn = np.loadtxt('./data/processed/weight/index_' + str(index_u_l_predict) + '_y_rn_Proposed.txt')

# creating a fake data as example
time = np.linspace(0, 23, 24)
print(time.shape)
print(time)

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
plt.title('Unit 1')
plt.legend()
plt.grid(True)
plt.show()
