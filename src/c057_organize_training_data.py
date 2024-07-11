## Organize training data for the surrogate model of optimizing weight

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from utils.case import Case
from utils.io import IO

n_components = 17
type_u_l = 'train'
folder_outputs = './data/processed/weight/outputs/'
parameter = Case().case_ieee30_parameter()

matrix_load, matrix_wind, matrix_weight, matrix_cost, matrix_cost_normalized = IO().read_training_data(type_u_l, u_select=parameter['u_select'], name_method='Proposed', folder_outputs=folder_outputs)

X = np.concatenate((matrix_load, matrix_wind), axis=0).T 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) # X_scaled = (X - scaler.mean_) / scaler.scale_
pca = PCA(n_components=n_components)
principalComponents = pca.fit_transform(X_scaled) # Components = pca.components_ * X_scaled
print('Variance: {}'.format(np.sum(pca.explained_variance_ratio_)))

np.save(folder_outputs + 'scaler_mean.npy', scaler.mean_)
np.save(folder_outputs + 'scaler_scale.npy', scaler.scale_)
np.save(folder_outputs + 'pca_components.npy', pca.components_)

print(IO().to_pca(X, folder_outputs=folder_outputs).shape)
