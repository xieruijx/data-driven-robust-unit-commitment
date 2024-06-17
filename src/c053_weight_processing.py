## Aggregate the cost results of different weights

import numpy as np
import matplotlib.pyplot as plt

from utils.io import IO

## Settings
index_u_l_predict = 16
type_u_l = 'test'

df, best_index, best_weight, best_cost = IO().read_weight_cost(index_u_l_predict, type_u_l, epsilon=0.05, name_method='Proposed', folder_number='./data/processed/weight/', folder_outputs='./data/processed/weight/outputs/', folder_strategies='./data/processed/weight/strategies/')

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
p1 = ax1.scatter(df['w0'], df['w1'], df['cost'], c=df['cost'], cmap=plt.cm.rainbow)
fig1.colorbar(p1)
ax1.set_xlabel('weight 1')
ax1.set_ylabel('weight 2')
ax1.set_zlabel('validation cost rank')

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
p2 = ax2.scatter(df['w0'], df['w1'], c=df['cost'], cmap=plt.cm.rainbow)
ax2.plot([0, 0], [0, 1], color='k')
ax2.plot([1, 0], [0, 0], color='k')
ax2.plot([1, 0], [0, 1], color='k')
fig2.colorbar(p2)
ax2.set_xlabel('weight 1')
ax2.set_ylabel('weight 2')

# ax2.scatter(df['w0'], df['w1'], marker='x', s=150)
ax2.scatter(df['w0'][df['cost'] == np.min(df['cost'])], df['w1'][df['cost'] == np.min(df['cost'])], marker='+', s=150)

print(best_index)
print(best_weight)
print(best_cost)

plt.show()