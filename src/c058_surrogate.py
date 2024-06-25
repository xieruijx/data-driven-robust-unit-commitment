## Surrogate model

import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split

from utils.io import IO

class MLP(nn.Module):
    def __init__(self, input_size, num_layers, units_per_layer):
        super(MLP, self).__init__()

        layers = []
        for i in range(num_layers):
            if i == 0: # first layer
                layers.append(nn.Linear(input_size, units_per_layer))
            else: # hidden layers
                layers.append(nn.Linear(units_per_layer, units_per_layer))
            layers.append(nn.ReLU())  
        layers.append(nn.Linear(units_per_layer, 1)) # output layer

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x
    
num_layers = 5
units_per_layer = 40
num_epoch = 50
folder_outputs = './data/processed/weight/outputs/'

matrix_load = np.load(folder_outputs + 'matrix_load.npy')
matrix_wind = np.load(folder_outputs + 'matrix_wind.npy')
matrix_pca = IO().to_pca(np.concatenate((matrix_load, matrix_wind), axis=0).T, folder_outputs=folder_outputs)
matrix_weight = np.load(folder_outputs + 'matrix_weight.npy')
features = np.concatenate((matrix_pca, matrix_weight.T), axis=1)

# matrix_cost = np.load(folder_outputs + 'matrix_cost.npy').reshape((-1, 1))
# features = features[matrix_cost[:, 0] < float('inf'), :]
# matrix_cost = matrix_cost[matrix_cost[:, 0] < float('inf'), :]

matrix_cost_normalized = np.load(folder_outputs + 'matrix_cost_normalized.npy').reshape((-1, 1))
features = features[matrix_cost_normalized[:, 0] < float('inf'), :]
print(np.max(np.max(features)))
print(np.min(np.min(features)))
matrix_cost = matrix_cost_normalized[matrix_cost_normalized[:, 0] < float('inf'), :]

# X_train, X_test, y_train, y_test = train_test_split(features, matrix_cost_normalized, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(features, matrix_cost, test_size=0.2, random_state=42)
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()
    
model = MLP(input_size=X_train.shape[1], num_layers=num_layers, units_per_layer=units_per_layer)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epoch = 50
for epoch in range(num_epoch):
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('Epoch {}: Loss: {}'.format(epoch, loss.item()))

with torch.no_grad():
    y_pred = model(X_test)
    test_loss = criterion(y_pred, y_test)

print("After {} epochs, Loss: {}".format(num_epoch, test_loss.item()))
