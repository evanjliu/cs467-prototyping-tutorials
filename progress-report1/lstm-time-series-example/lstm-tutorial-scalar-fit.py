import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from sklearn.preprocessing import MinMaxScaler
 
df = pd.read_csv('airline-passengers.csv')
scaler = MinMaxScaler(feature_range=(0, 1))
timeseries = scaler.fit_transform(df[["Passengers"]].values.astype('float32'))

# train-test split for time series
train_size = int(len(timeseries) * 0.67)
test_size = len(timeseries) - train_size
train, test = timeseries[:train_size], timeseries[train_size:]
 
def create_dataset(dataset, lookback):
    """Transform a time series into a prediction dataset
    
    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback]
        target = dataset[i+1:i+lookback+1]
        X.append(feature)
        y.append(target)
    return torch.tensor(X), torch.tensor(y)
 
lookback = 4
X_train, y_train = create_dataset(train, lookback=lookback)
X_test, y_test = create_dataset(test, lookback=lookback)
 
class AirModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x
 
model = AirModel()
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)
 
n_epochs = 2000
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    if epoch % 100 != 0:
        continue
    model.eval()
    with torch.no_grad():
        y_pred = model(X_train)
        train_rmse = np.sqrt(loss_fn(y_pred, y_train))
        y_pred = model(X_test)
        test_rmse = np.sqrt(loss_fn(y_pred, y_test))
    print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))
 
with torch.no_grad():
    # Convert predictions back to NumPy
    y_train_pred = model(X_train).detach().numpy()
    y_test_pred = model(X_test).detach().numpy()

    # Only take the last prediction in each sequence
    y_train_pred = y_train_pred[:, -1, :]
    y_test_pred = y_test_pred[:, -1, :]

    # Convert back to original scale
    y_train_pred = scaler.inverse_transform(y_train_pred.reshape(-1, 1))
    y_test_pred = scaler.inverse_transform(y_test_pred.reshape(-1, 1))

    # Shift train predictions for plotting
    train_plot = np.ones_like(timeseries) * np.nan  
    train_plot = train_plot.flatten()  
    train_plot[lookback:train_size] = y_train_pred.flatten()  

    # Shift test predictions for plotting
    test_plot = np.ones_like(timeseries) * np.nan
    test_plot = test_plot.flatten()  
    test_plot[train_size+lookback:len(timeseries)] = y_test_pred.flatten()
    
# plot
plt.figure(figsize=(10,5))
plt.plot(scaler.inverse_transform(timeseries), label="Actual Data", color='blue')
plt.plot(train_plot, label="Train Predictions", color='red')
plt.plot(test_plot, label="Test Predictions", color='green')
plt.legend()
plt.show()
