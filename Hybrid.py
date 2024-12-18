import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
ARIMA_WEIGHT = 0.7  # Weight for ARIMA predictions in hybrid model



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


# Transformer model
class Transformer_Price(nn.Module):
    def __init__(self, input_dim, embed_dim, nhead, num_layers, output_dim, dropout=0.1):
        super(Transformer_Price, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, 1000, embed_dim))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dropout=dropout),
            num_layers=num_layers
        )
        self.fc = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = x + self.positional_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        x = self.fc(x[:, -1, :])
        return x


# ARIMA prediction function
def ARIMA_Price(data, start_idx, num_predictions):
    try:
        # Fit ARIMA model with fixed parameters
        model = ARIMA(data[:start_idx], order=(1,1,1))
        fitted_model = model.fit()
        
        # Make predictions
        predictions = fitted_model.forecast(steps=num_predictions)
        return predictions
    except:
        # Return last value if ARIMA fails
        return np.array([data[start_idx-1]] * num_predictions)


# Train the hybrid model
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



def metrics(y_true, y_pred, model_name):

    # Error-based metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # MAPE calculation (avoiding division by zero)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    # R² and Adjusted R²
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    p = 1  # number of predictors (features)
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    # SMAPE calculation
    smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100
    
    # Print metrics
    print(f"\n {model_name} Performance Metrics:")
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



# Fetch and preprocess data
hist = stock_data(TICKER)
hist = hist.reset_index()
hist = hist[['Date', 'Close']]


# Store original data for ARIMA
original_data = hist['Close'].values


# Normalize data for Transformer
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(hist[['Close']])


# Prepare data for Transformer
X, y = [], []
for i in range(len(scaled_data) - SEQ_LENGTH):
    X.append(scaled_data[i:i + SEQ_LENGTH])
    y.append(scaled_data[i + SEQ_LENGTH])
X, y = np.array(X), np.array(y)

train_size = int(0.9 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


# Create datasets and loaders
train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


# Initialize and train Transformer model
model = Transformer_Price(input_dim=X.shape[2], embed_dim=EMBED_DIM, nhead=NHEAD,
                        num_layers=NUM_LAYERS, output_dim=OUTPUT_DIM)
train_model(model, train_loader, test_loader, EPOCHS, LEARNING_RATE)


# Generate hybrid predictions
model.eval()
transformer_predictions = []
hybrid_predictions = []

with torch.no_grad():
    current_sequence = torch.FloatTensor(scaled_data[:SEQ_LENGTH]).unsqueeze(0)
    
    for i in range(len(scaled_data) - SEQ_LENGTH):
        # Transformer prediction
        transformer_pred = model(current_sequence).numpy()[0]
        
        # ARIMA prediction
        arima_pred = ARIMA_Price(original_data[:SEQ_LENGTH+i], SEQ_LENGTH+i, 1)[0]
        arima_pred = scaler.transform([[arima_pred]])[0]
        
        # Combine predictions
        hybrid_pred = (1 - ARIMA_WEIGHT) * transformer_pred + ARIMA_WEIGHT * arima_pred
        
        transformer_predictions.append(transformer_pred)
        hybrid_predictions.append(hybrid_pred)
        
        if i < len(scaled_data) - SEQ_LENGTH - 1:
            current_sequence = torch.FloatTensor(scaled_data[i+1:i+1+SEQ_LENGTH]).unsqueeze(0)


# Generate future predictions
transformer_future = []
hybrid_future = []

last_sequence = scaled_data[-SEQ_LENGTH:].reshape(1, SEQ_LENGTH, 1)
last_sequence = torch.FloatTensor(last_sequence)

with torch.no_grad():
    current_sequence = last_sequence
    for step in range(FUTURE_STEPS):
        # Transformer prediction
        transformer_pred = model(current_sequence).numpy()[0]
        
        # ARIMA prediction
        arima_pred = ARIMA_Price(original_data, len(original_data) + step, 1)[0]
        arima_pred = scaler.transform([[arima_pred]])[0]
        
        # Combine predictions
        hybrid_pred = (1 - ARIMA_WEIGHT) * transformer_pred + ARIMA_WEIGHT * arima_pred
        
        transformer_future.append(transformer_pred)
        hybrid_future.append(hybrid_pred)
        
        new_sequence = current_sequence.numpy()[:, 1:, :]
        new_point = hybrid_pred.reshape(1, 1, 1)
        current_sequence = torch.FloatTensor(np.concatenate([new_sequence, new_point], axis=1))


# Inverse transform predictions
transformer_predictions = scaler.inverse_transform(np.array(transformer_predictions).reshape(-1, 1))
hybrid_predictions = scaler.inverse_transform(np.array(hybrid_predictions).reshape(-1, 1))
transformer_future = scaler.inverse_transform(np.array(transformer_future).reshape(-1, 1))
hybrid_future = scaler.inverse_transform(np.array(hybrid_future).reshape(-1, 1))


# Calculate metrics for both transformer and hybrid predictions
actual_values = hist['Close'][SEQ_LENGTH:].values
transformer_metrics = metrics(actual_values, transformer_predictions.flatten(), "Transformer")
hybrid_metrics = metrics(actual_values, hybrid_predictions.flatten(), "Hybrid Model")


# Plot results
fig = go.Figure()

pred_dates = hist['Date'][SEQ_LENGTH:]
future_dates = pd.date_range(start=hist['Date'].iloc[-1], periods=FUTURE_STEPS+1)[1:]

fig.add_trace(go.Scatter(x=hist['Date'], y=hist['Close'], mode='lines', name='Observed'))
fig.add_trace(go.Scatter(x=pred_dates, y=transformer_predictions.flatten(), mode='lines', 
                        name='Transformer Predictions', line=dict(color='red')))
fig.add_trace(go.Scatter(x=pred_dates, y=hybrid_predictions.flatten(), mode='lines', 
                        name='Hybrid Predictions', line=dict(color='purple')))
fig.add_trace(go.Scatter(x=future_dates, y=hybrid_future.flatten(), mode='lines',
                        name='Hybrid Forecast', line=dict(color='green')))

fig.update_layout(
    autosize=False,
    width=800,
    height=500,
    xaxis_title='Date',
    yaxis_title='Price $',
    title=f'{TICKER} Hybrid Model (Transformer + ARIMA) Predictions and Forecast'
)
fig.show()