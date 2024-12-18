import numpy as np
import pandas as pd
import yfinance as yf
import xgboost as xgb
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')



# Parameters
TICKER = 'MSFT' 
PERIOD = '2y'       # Data period for stock prices
SEQ_LENGTH = 15     # Number of time steps to look back
FUTURE_STEPS = 10   # Number of future predictions


# XGBoost specific parameters
xgb_params = {
    'objective': 'reg:squarederror',      # Regression task
    'learning_rate': 0.01,               # Learning rate
    'max_depth': 5,                     # Maximum depth of a tree
    'n_estimators': 1000,              # Number of trees
    'min_child_weight': 1,            # Minimum sum of instance weight needed in a child
    'subsample': 0.8,                # Subsample ratio of the training instances
    'colsample_bytree': 0.8,        # Subsample ratio of columns when constructing each tree
    'gamma': 0,                    # Minimum loss reduction required to make a further partition on a leaf node of the tree
    'eta': 0.1,                   # Step size shrinkage used in update to prevent overfitting
    'seed': 42                   # Random seed
}



# Importing stock price data using yfinance with given ticker and period
def stock_data(ticker):
    stock_data = yf.Ticker(ticker)
    return stock_data.history(PERIOD)


# Create sequences from time series data
def create_seq(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


# Prepare data for XGBoost (flatten sequences)
def data_xgb(X):
    # Reshape from (n_samples, seq_length, n_features) to (n_samples, seq_length * n_features)
    return X.reshape(X.shape[0], -1)


# Calculate percentage errors
def pct_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # MAPE calculation
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # SMAPE calculation
    smape = np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100
    
    return mape, smape



# Fetch and preprocess data
hist = stock_data(TICKER)
hist = hist.reset_index()
hist = hist[['Date', 'Close']]


# Normalize data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(hist[['Close']])


# Create sequences
X, y = create_seq(scaled_data, SEQ_LENGTH)

# Prepare data for XGBoost
X_xgb = data_xgb(X)

# Train/test split
train_size = int(0.95 * len(X_xgb)) # 95% training data
X_train, X_test = X_xgb[:train_size], X_xgb[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


# Initialize and train XGBoost model
model = xgb.XGBRegressor(**xgb_params)

# Train with early stopping
model.fit(
    X_train, y_train,
    eval_set = [(X_test, y_test)],
    verbose=False
)


# Generate predictions for historical data
all_predictions = []

for i in range(len(X_xgb)):
    pred = model.predict(X_xgb[i:i+1])[0]
    all_predictions.append(pred)

# Inverse transform predictions
all_predictions = scaler.inverse_transform(np.array(all_predictions).reshape(-1, 1))


# Generate future predictions
future_predictions = []
last_sequence = scaled_data[-SEQ_LENGTH:].reshape(1, -1)

for _ in range(FUTURE_STEPS):
    # Make prediction
    prediction = model.predict(last_sequence)[0]
    future_predictions.append(prediction)
    
    # Update sequence
    last_sequence = last_sequence.flatten()
    last_sequence = np.roll(last_sequence, -1)
    last_sequence[-1] = prediction
    last_sequence = last_sequence.reshape(1, -1)

# Inverse transform future predictions
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))


# Generate predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)


# Inverse transform predictions and actual values for meaningful metrics
y_train_orig = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
train_pred_orig = scaler.inverse_transform(train_predictions.reshape(-1, 1)).flatten()
test_pred_orig = scaler.inverse_transform(test_predictions.reshape(-1, 1)).flatten()


# Calculate metrics for test set
mae = mean_absolute_error(y_test_orig, test_pred_orig)
mse = mean_squared_error(y_test_orig, test_pred_orig)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_orig, test_pred_orig)

# Calculate adjusted R² (n = number of samples, p = number of features)
n = len(y_test_orig)
p = SEQ_LENGTH  # number of features (time steps used for prediction)
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
future_dates = pd.date_range(start = hist['Date'].iloc[-1], periods = FUTURE_STEPS+1)[1:]

fig.add_trace(go.Scatter(x=hist['Date'], y=hist['Close'], mode='lines', 
                        name='Observed'))
fig.add_trace(go.Scatter(x=pred_dates, y=all_predictions.flatten(), mode='lines', 
                        name='XGBoost Predictions', line=dict(color='red')))
fig.add_trace(go.Scatter(x=future_dates, y=future_predictions.flatten(), mode='lines',
                        name='XGBoost Forecast', line=dict(color='green')))

fig.update_layout(
    autosize=False,
    width=800,
    height=500,
    xaxis_title='Date',
    yaxis_title='Price $',
    title=f'{TICKER} XGBoost Model Predictions and Forecast'
)
fig.show()