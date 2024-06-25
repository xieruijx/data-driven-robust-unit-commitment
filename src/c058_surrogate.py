## Surrogate model

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils.io import IO

class MLP(nn.Module):
    def __init__(self, input_size, num_layers, units_per_layer, dropout_rate):
        super(MLP, self).__init__()

        layers = []
        for i in range(num_layers):
            if i == 0: # first layer
                layers.append(nn.Linear(input_size, units_per_layer))
            else: # hidden layers
                layers.append(nn.Linear(units_per_layer, units_per_layer))
            layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.ReLU())  
        layers.append(nn.Linear(units_per_layer, 1)) # output layer

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x
    
num_layers = 2
units_per_layer = 16
weight_decay = 0.02
num_pca = 3
dropout_rate = 0
folder_outputs = './data/processed/weight/outputs/'
num_epoch = 4000

matrix_load = np.load(folder_outputs + 'matrix_load.npy')
matrix_wind = np.load(folder_outputs + 'matrix_wind.npy')
matrix_pca = IO().to_pca(np.concatenate((matrix_load, matrix_wind), axis=0).T, folder_outputs=folder_outputs)
matrix_pca = matrix_pca[:, :num_pca]
matrix_weight = np.load(folder_outputs + 'matrix_weight.npy')
X = np.concatenate((matrix_pca, matrix_weight.T), axis=1)

matrix_cost = np.load(folder_outputs + 'matrix_cost.npy').reshape((-1, 1))
X = X[matrix_cost[:, 0] < float('inf'), :]
y = matrix_cost[matrix_cost[:, 0] < float('inf'), :]

# matrix_cost_normalized = np.load(folder_outputs + 'matrix_cost_normalized.npy').reshape((-1, 1))
# X = X[matrix_cost_normalized[:, 0] < float('inf'), :]
# y = matrix_cost_normalized[matrix_cost_normalized[:, 0] < float('inf'), :]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_scaler = StandardScaler()
X_train = X_scaler.fit_transform(X_train)
X_test = X_scaler.transform(X_test)
y_scaler = StandardScaler()
y_train = y_scaler.fit_transform(y_train)
y_test = y_scaler.transform(y_test)

X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()
    
model = MLP(input_size=X_train.shape[1], num_layers=num_layers, units_per_layer=units_per_layer, dropout_rate=dropout_rate)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)

curve_loss = np.zeros((num_epoch, 2))

for epoch in range(num_epoch):
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    curve_loss[epoch, 0] = loss.item()

    with torch.no_grad():
        y_pred = model(X_test)
        test_loss = criterion(y_pred, y_test)
    curve_loss[epoch, 1] = test_loss.item()

# Draw the picture of training model
fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True, figsize=(9, 5))
axs[0].plot(range(num_epoch), curve_loss[:, 0])
axs[1].plot(range(num_epoch), curve_loss[:, 1])

# Output the model parameters
param_dict = model.state_dict()
param_model = {}
param_model['num_layers'] = num_layers
param_model['weight0'] = param_dict['layers.0.weight'].cpu().numpy()
param_model['bias0'] = param_dict['layers.0.bias'].cpu().numpy()
param_model['weight1'] = param_dict['layers.3.weight'].cpu().numpy()
param_model['bias1'] = param_dict['layers.3.bias'].cpu().numpy()
param_model['weight2'] = param_dict['layers.6.weight'].cpu().numpy()
param_model['bias2'] = param_dict['layers.6.bias'].cpu().numpy()
IO().output_param_model(param_model, folder_outputs)

plt.show()
