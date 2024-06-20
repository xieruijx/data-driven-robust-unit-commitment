## Analyze the data

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

folder_outputs = './data/processed/weight/outputs/'
matrix_load = np.load(folder_outputs + 'matrix_load.npy')
matrix_wind = np.load(folder_outputs + 'matrix_wind.npy')
matrix_weight = np.load(folder_outputs + 'matrix_weight.npy')
matrix_cost = np.load(folder_outputs + 'matrix_cost.npy')

X = np.concatenate((matrix_load, matrix_wind), axis=0).T # n_components=39 for 0.99; 30, 0.98; 24, 0.97; 17, 0.95; 8, 0.90
# X = matrix_load.T # n_components=38 for 0.99
# X = matrix_wind.T # n_components=9 for 0.99
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) # X_scaled = (X - scaler.mean_) / scaler.scale_
pca = PCA()
principalComponents = pca.fit_transform(X_scaled) # Components = pca.components_ * X_scaled

np.save(folder_outputs + 'scaler_mean.npy', scaler.mean_)
np.save(folder_outputs + 'scaler_scale.npy', scaler.scale_)
np.save(folder_outputs + 'pca_components.npy', pca.components_)
