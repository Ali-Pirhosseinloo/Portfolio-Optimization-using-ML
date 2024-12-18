import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')



# Parameters
TICKER = 'MSFT' 
PERIOD = '2y'       # Data period for stock prices
SEQ_LENGTH = 15     # Number of time steps to look back
FUTURE_STEPS = 10   # Number of future predictions

# SVR parameters for grid search
param_grid = {
    'kernel': ['rbf', 'linear'],              # Kernel type 
    'C': [0.1, 1, 10, 100],                   # Regularization parameter
    'epsilon': [0.01, 0.1, 0.2],              # Epsilon insensitive loss function
    'gamma': ['scale', 'auto', 0.1, 0.01]     # Kernel coefficient
}



# Importing stock price data using yfinance with given ticker and period
def stock_data(ticker):
    stock_data = yf.Ticker(ticker)
    hist = stock_data.history(PERIOD)
    hist = hist.reset_index()[['Date', 'Close']]
    
    # Add technical indicators
    hist['SMA_5'] = hist['Close'].rolling(window=5).mean()
    hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
    hist['RSI'] = hist['Close'].diff().rolling(window=14).apply(
        lambda x: 100 - (100 / (1 + (x[x > 0].mean() / -x[x < 0].mean())))
    )
    
    return hist.fillna(method='bfill')



# Create sequences from time series data
def create_seq(data, seq_length):
    features = ['Close', 'SMA_5', 'SMA_20', 'RSI']
    scaled_data = scaler.fit_transform(data[features])
    X, y = [], []
    for i in range(len(scaled_data) - seq_length):
        X.append(scaled_data[i:i + seq_length])
        y.append(scaled_data[i + seq_length])
    
    X = np.array(X).reshape(len(X), -1)  # Flatten sequences
    y = np.array(y)[:, 0]  # We only predict the 'Close' price
    return X, y



# Calculate percentage errors
def pct_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # MAPE calculation
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # SMAPE calculation
    smape = np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100
    
    return mape, smape



# Function to generate future predictions
def generate_future_predictions(model, scaled_data, scaler):
    features = ['Close', 'SMA_5', 'SMA_20', 'RSI']
    future_predictions = []
    last_sequence = scaled_data[-SEQ_LENGTH:].reshape(1, -1)
    
    for _ in range(FUTURE_STEPS):
        prediction = model.predict(last_sequence)[0]
        
        next_features = np.zeros(len(features))
        next_features[0] = prediction
        next_features[1] = np.mean(last_sequence[0, -5:])
        next_features[2] = np.mean(last_sequence[0, -20:])
        next_features[3] = last_sequence[0, -1]
        
        future_predictions.append(prediction)
        last_sequence = np.roll(last_sequence, -len(features))
        last_sequence[0, -len(features):] = next_features
    
    predictions_scaled = np.zeros((len(future_predictions), len(features)))
    predictions_scaled[:, 0] = future_predictions
    return scaler.inverse_transform(predictions_scaled)[:, 0]



# Fetch and preprocess data
hist = stock_data(TICKER)
features = ['Close', 'SMA_5', 'SMA_20', 'RSI']


# Initialize scaler
scaler = MinMaxScaler()


# Create sequences
X, y = create_seq(hist, SEQ_LENGTH)


# Train/test split
train_size = int(0.95 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


# Initialize and optimize SVR model
grid_search = GridSearchCV(
    SVR(), param_grid, cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1, verbose=1
)

grid_search.fit(X_train, y_train)
print("\nBest parameters found:", grid_search.best_params_)
print("Best MSE score:", -grid_search.best_score_)


# Get the best model
model = grid_search.best_estimator_


# Generate predictions for historical data
all_predictions = []
for i in range(len(X)):
    pred = model.predict(X[i:i+1])[0]
    all_predictions.append(pred)


# Prepare for inverse transform
predictions_scaled = np.zeros((len(all_predictions), len(features)))
predictions_scaled[:, 0] = all_predictions
all_predictions = scaler.inverse_transform(predictions_scaled)[:, 0]


# Generate future predictions
scaled_data = scaler.transform(hist[features])
future_predictions = generate_future_predictions(model, scaled_data, scaler)


# Generate predictions for metrics
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)


# Prepare for inverse transform
train_pred_scaled = np.zeros((len(train_predictions), len(features)))
test_pred_scaled = np.zeros((len(test_predictions), len(features)))
train_pred_scaled[:, 0] = train_predictions
test_pred_scaled[:, 0] = test_predictions


# Inverse transform predictions for meaningful metrics
train_pred_orig = scaler.inverse_transform(train_pred_scaled)[:, 0]
test_pred_orig = scaler.inverse_transform(test_pred_scaled)[:, 0]


# Prepare true values for metrics
y_train_scaled = np.zeros((len(y_train), len(features)))
y_test_scaled = np.zeros((len(y_test), len(features)))
y_train_scaled[:, 0] = y_train
y_test_scaled[:, 0] = y_test

y_train_orig = scaler.inverse_transform(y_train_scaled)[:, 0]
y_test_orig = scaler.inverse_transform(y_test_scaled)[:, 0]


# Calculate metrics for test set
mae = mean_absolute_error(y_test_orig, test_pred_orig)
mse = mean_squared_error(y_test_orig, test_pred_orig)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_orig, test_pred_orig)

# Calculate adjusted R² (n = number of samples, p = number of features)
n = len(y_test_orig)
p = SEQ_LENGTH * len(features)  # number of features
adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

# Calculate MAPE and SMAPE
mape, smape = pct_error(y_test_orig, test_pred_orig)


# Print metrics
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


# Plot results
fig = go.Figure()

pred_dates = hist['Date'][SEQ_LENGTH:]
future_dates = pd.date_range(start=hist['Date'].iloc[-1], periods=FUTURE_STEPS+1)[1:]

fig.add_trace(go.Scatter(x=hist['Date'], y=hist['Close'], mode='lines', name='Observed'))
fig.add_trace(go.Scatter(x=pred_dates, y=all_predictions, mode='lines', 
                        name='SVR Predictions', line=dict(color='red')))
fig.add_trace(go.Scatter(x=future_dates, y=future_predictions, mode='lines',
                        name='SVR Forecast', line=dict(color='green')))

fig.update_layout(
    autosize=False, width=800, height=500,
    xaxis_title='Date', yaxis_title='Price $',
    title=f'{TICKER} SVR Model Predictions and Forecast'
)
fig.show()