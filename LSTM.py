import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')



# Model parameters
TICKER = 'MSFT'
PERIOD = '2y'       # Data period for stock prices
SEQ_LENGTH = 15     # Number of time steps to look back
FUTURE_STEPS = 10   # Number of future predictions
BATCH_SIZE = 32     # Number of samples per batch
HIDDEN_DIM = 64  
NUM_LAYERS = 2
OUTPUT_DIM = 1
EPOCHS = 100
LEARNING_RATE = 0.001



# Importing stock price data using yfinance with given ticker and period
def stock_data(ticker):
    stock_data = yf.Ticker(ticker)
    return stock_data.history(PERIOD)


# Dataset class for time series
class TimeSeriesDataset(Dataset):

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



# Neural Network (LSTM) model
class LSTM_Prediction(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout = 0.2):
        super(LSTM_Prediction, self).__init__()
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential( 
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        # LSTM layer
        lstm_out, _ = self.lstm(x)
        
        # Take the last time step's output
        last_time_step = lstm_out[:, -1, :]
        
        # Pass through fully connected layers
        output = self.fc_layers(last_time_step)
        return output
    


def metrics(y_true, y_pred, n_features=1):

    # Error-based metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # Percentage-based metrics
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Scale-independent metrics
    r2 = r2_score(y_true, y_pred)
    
    # Adjusted R² calculation
    n = len(y_true)
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    
    # Symmetric Mean Absolute Percentage Error
    smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100
    
    print("\nModel Performance Metrics:")
    print("-" * 50)
    print(f"Error-based metrics:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.4f}")
    print(f"\nScale-independent metrics:")
    print(f"R²: {r2:.4f}")
    print(f"Adjusted R²: {adjusted_r2:.4f}")
    print(f"SMAPE: {smape:.4f}\n")
    
    return {
        'mae': mae, 'mse': mse, 'rmse': rmse, 'mape': mape,
        'r2': r2, 'adjusted_r2': adjusted_r2, 'smape': smape
    }



# Train the Neural Network model
def train_model(model, train_loader, val_loader, epochs, lr):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")



# Fetch and preprocess data
hist = stock_data(TICKER)
hist = hist.reset_index()
hist = hist[['Date', 'Close']]
hist['Close'] = hist['Close'].interpolate()


# Normalize data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(hist[['Close']])


# Data preparation
X, y = [], []
for i in range(len(scaled_data) - SEQ_LENGTH):
    X.append(scaled_data[i:i + SEQ_LENGTH])
    y.append(scaled_data[i + SEQ_LENGTH])
X, y = np.array(X), np.array(y)


# Train/test split
train_size = int(0.95 * len(X)) # 95% training data
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


# Create datasets and loaders
train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


# Initialize and train model
model = LSTM_Prediction(
    input_dim=X.shape[2],
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    output_dim=OUTPUT_DIM
)
train_model(model, train_loader, test_loader, EPOCHS, LEARNING_RATE)


# Calculate metrics on test set
model.eval()
y_true_test = []
y_pred_test = []


with torch.no_grad():
    for X_batch, y_batch in test_loader:
        predictions = model(X_batch)
        y_true_test.extend(y_batch.numpy())
        y_pred_test.extend(predictions.numpy())


# Convert to numpy arrays and inverse transform
y_true_test = scaler.inverse_transform(np.array(y_true_test))
y_pred_test = scaler.inverse_transform(np.array(y_pred_test))


# Calculate and print metrics
metrics = metrics(y_true_test, y_pred_test)


# Generate predictions using rolling window approach
all_predictions = []

with torch.no_grad():
    # Initial sequence
    current_sequence = torch.FloatTensor(scaled_data[:SEQ_LENGTH]).unsqueeze(0)
    
    for i in range(len(scaled_data) - SEQ_LENGTH):
        # Predict next value
        pred = model(current_sequence).numpy()[0]
        all_predictions.append(pred)
        
        # Update sequence with actual value for next prediction
        if i < len(scaled_data) - SEQ_LENGTH - 1:
            current_sequence = torch.FloatTensor(scaled_data[i+1:i+1+SEQ_LENGTH]).unsqueeze(0)


# Inverse transform predictions
all_predictions = scaler.inverse_transform(np.array(all_predictions).reshape(-1, 1))

# Generate future predictions
last_sequence = scaled_data[-SEQ_LENGTH:].reshape(1, SEQ_LENGTH, 1)
last_sequence = torch.FloatTensor(last_sequence)
future_predictions = []


with torch.no_grad():
    current_sequence = last_sequence
    for _ in range(FUTURE_STEPS):
        prediction = model(current_sequence).numpy()
        future_predictions.append(prediction[0])
        new_sequence = current_sequence.numpy()[:, 1:, :]
        new_point = prediction.reshape(1, 1, 1)
        current_sequence = torch.FloatTensor(np.concatenate([new_sequence, new_point], axis=1))

future_predictions = scaler.inverse_transform(np.array(future_predictions))


# Plot results
fig = go.Figure()

pred_dates = hist['Date'][SEQ_LENGTH:]
future_dates = pd.date_range(start=hist['Date'].iloc[-1], periods=FUTURE_STEPS+1)[1:]

fig.add_trace(go.Scatter(x=hist['Date'], y=hist['Close'], mode='lines', name='Observed'))
fig.add_trace(go.Scatter(x=pred_dates, y=all_predictions.flatten(), mode='lines', 
                        name='Neural Network Predictions', line=dict(color='red')))
fig.add_trace(go.Scatter(x=future_dates, y=future_predictions.flatten(), mode='lines',
                        name='Neural Network (LSTM) Forecast', line=dict(color='green')))

fig.update_layout(
    autosize=False,
    width=800,
    height=500,
    xaxis_title='Date',
    yaxis_title='Price $',
    title=f'{TICKER} Neural Network (LSTM) Model Predictions and Forecast'
)
fig.show()