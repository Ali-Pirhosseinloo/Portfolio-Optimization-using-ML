import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
import plotly.graph_objects as go
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


# Model Parameters
TICKER = 'MSFT'
PERIOD = '2y'       # Data period for stock prices
SEQ_LENGTH = 15     # Number of time steps to look back
FUTURE_STEPS = 10   # Number of future predictions
BATCH_SIZE = 32
EMBED_DIM = 16 
NHEAD = 4
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
        self.X = torch.tensor(X, dtype = torch.float32)
        self.y = torch.tensor(y, dtype = torch.float32)
    
    def __len__(self): 
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



# Transformer model
class Transformer_Price(nn.Module):
    def __init__(self, input_dim, embed_dim, nhead, num_layers, output_dim, dropout=0.1):
        super(Transformer_Price, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, 1000, embed_dim))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dropout=dropout),
            num_layers = num_layers
        )
        self.fc = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = x + self.positional_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        x = self.fc(x[:, -1, :])  # Use the last time step's representation
        return x



# Train the Transformer model
def train_model(model, train_loader, val_loader, epochs, lr):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
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

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")



# Function to calculate performance metrics
def metrics(y_true, y_pred):

    metrics = {}
    
    # Error-based metrics
    metrics['MAE'] = mean_absolute_error(y_true, y_pred)
    metrics['MSE'] = mean_squared_error(y_true, y_pred)
    metrics['RMSE'] = np.sqrt(metrics['MSE'])
    
    # Percentage-based metrics
    # Avoid division by zero in MAPE calculation
    non_zero_mask = y_true != 0
    metrics['MAPE'] = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
    
    # SMAPE calculation
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    non_zero_denominator = denominator != 0
    metrics['SMAPE'] = np.mean(np.abs(y_pred[non_zero_denominator] - y_true[non_zero_denominator]) / denominator[non_zero_denominator]) * 100
    
    # Scale-independent metrics
    metrics['R2'] = r2_score(y_true, y_pred)
    
    # Adjusted R-squared
    n = len(y_true)
    p = 1  # number of predictors (features)
    metrics['Adjusted_R2'] = 1 - (1 - metrics['R2']) * (n - 1) / (n - p - 1)
    
    return metrics


# Fetch and preprocess data
hist = stock_data(TICKER)
hist = hist.reset_index()
hist = hist[['Date', 'Close']]
hist['Close'] = hist['Close'].interpolate()


# Normalize data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(hist[['Close']])


# Prepare data for training
X, y = [], []
for i in range(len(scaled_data) - SEQ_LENGTH):
    X.append(scaled_data[i:i + SEQ_LENGTH])
    y.append(scaled_data[i + SEQ_LENGTH])
X, y = np.array(X), np.array(y)


# Train/test split
train_size = int(0.9 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


# Create datasets and loaders
train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)  # Enable shuffle
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


# Initialize and train model
model = Transformer_Price(input_dim=X.shape[2], embed_dim=EMBED_DIM, nhead=NHEAD,
                        num_layers=NUM_LAYERS, output_dim=OUTPUT_DIM)
train_model(model, train_loader, test_loader, EPOCHS, LEARNING_RATE)


# Generate predictions using rolling window approach
model.eval()
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


# Calculate metrics for the historical predictions
actual_values = hist['Close'][SEQ_LENGTH:].values
historical_metrics = metrics(actual_values, all_predictions.flatten())

# Print metrics in a formatted way
print("\nModel Performance Metrics:")
print("-" * 50)
print(f"Error-based metrics:")
print(f"MAE: {historical_metrics['MAE']:.4f}")
print(f"MSE: {historical_metrics['MSE']:.4f}")
print(f"RMSE: {historical_metrics['RMSE']:.4f}")
print(f"MAPE: {historical_metrics['MAPE']:.4f}%")
print(f"\nScale-independent metrics:")
print(f"R²: {historical_metrics['R2']:.4f}")
print(f"Adjusted R²: {historical_metrics['Adjusted_R2']:.4f}")
print(f"SMAPE: {historical_metrics['SMAPE']:.4f}%\n")


# Plot results
fig = go.Figure()

pred_dates = hist['Date'][SEQ_LENGTH:]
future_dates = pd.date_range(start=hist['Date'].iloc[-1], periods=FUTURE_STEPS+1)[1:]

fig.add_trace(go.Scatter(x=hist['Date'], y=hist['Close'], mode='lines', name='Observed'))
fig.add_trace(go.Scatter(x=pred_dates, y=all_predictions.flatten(), mode='lines', 
                        name='Transformer Predictions', line=dict(color='red')))
fig.add_trace(go.Scatter(x=future_dates, y=future_predictions.flatten(), mode='lines',
                        name='Transformer Forecast', line=dict(color='green')))

fig.update_layout(
    autosize=False,
    width=800,
    height=500,
    xaxis_title='Date',
    yaxis_title='Price $',
    title=f'{TICKER} Transformer Model Predictions and Forecast'
)
fig.show()