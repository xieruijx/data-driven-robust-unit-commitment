## Use the surrogate model to optimize the weight

import numpy as np
import matplotlib.pyplot as plt

from utils.case import Case
from utils.io import IO

## Settings
index_u_l_predict = 16
type_u_l = 'test'
num_points = 50
low = 0
high = 1
num_pca = 3
folder_outputs = './data/processed/weight/outputs/'
parameter = Case().case_ieee30_parameter()

weight_optimize = np.loadtxt('./data/processed/combination/d053_weight.txt')

X_scaler_mean = np.load(folder_outputs + 'X_scaler_mean.npy')
X_scaler_scale = np.load(folder_outputs + 'X_scaler_scale.npy')
y_scaler_mean = np.load(folder_outputs + 'y_scaler_mean.npy')
y_scaler_scale = np.load(folder_outputs + 'y_scaler_scale.npy')

predict_load, predict_wind = IO().read_test_sample(index_u_l_predict, type_u_l, u_select=parameter['u_select'])
input_pca = IO().to_pca(np.concatenate((predict_load, predict_wind), axis=0).reshape((1, -1)), folder_outputs=folder_outputs).reshape((-1,))

param_model = IO().read_param_model(folder_outputs=folder_outputs)

w0 = np.ones((num_points, 1)) @ (np.linspace(0, 1, num=num_points).reshape((1, -1)))
w1 = np.zeros((num_points, num_points))
for i in range(num_points):
    w1[:, i] = np.linspace(0, - i / (num_points - 1) + 1, num=num_points)
c = np.zeros((num_points, num_points))
for i in range(num_points):
    for j in range(num_points):
        input = (np.append(np.append(input_pca[:num_pca], w0[i, j]), w1[i, j]) - X_scaler_mean) / X_scaler_scale
        cost = IO().NN_ReLU(param_model=param_model, input=input).reshape(())
        c[i, j] = cost * y_scaler_scale[0] + y_scaler_mean[0]

z = c + 88700
labelsize = None
fontsize = 12

fig = plt.figure()
fig.set_figheight(4)
fig.set_figwidth(6)
ax1 = fig.add_subplot(121, projection='3d')
norm = plt.Normalize(c.min(), c.max())
colors = plt.cm.plasma(norm(c))
rcount, ccount, _ = colors.shape
surf = ax1.plot_surface(w0, w1, z, linewidth=0, antialiased=False, rcount=rcount, ccount=ccount, facecolors=colors, shade=False)
ax1.tick_params(axis='x', labelsize=labelsize)
ax1.tick_params(axis='y', labelsize=labelsize)
ax1.tick_params(axis='z', labelsize=labelsize)
ax1.set_xlabel('$w^{(0)}$', fontsize=fontsize)
ax1.set_ylabel('$w^{(1)}$', fontsize=fontsize)
ax1.set_zlabel('Total cost (\$)', fontsize=fontsize)

ax2 = fig.add_subplot(122)
ax2.fill_between([0, 1], [0, 0], [1, 0], label='$\mathcal{W}$',
                    color='gray', alpha=0.15)
cset = ax2.contour(w0, w1, z, cmap=plt.get_cmap('plasma'))
ax2.scatter(weight_optimize[0], weight_optimize[1], c='k', s=40, marker='*', label='Optimal weight')
ax2.tick_params(axis='x', labelsize=labelsize)
ax2.tick_params(axis='y', labelsize=labelsize)
ax2.set_xlabel('$w^{(0)}$', fontsize=fontsize)
ax2.set_ylabel('$w^{(1)}$', fontsize=fontsize)
ax2.clabel(cset, fontsize=labelsize)
ax2.legend()

plt.tight_layout()
plt.show()