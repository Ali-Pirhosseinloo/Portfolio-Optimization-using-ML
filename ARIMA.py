import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf, adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')



# Parameters
TICKER = 'MSFT'
PERIOD = '2y'           # Data period for stock prices
FORECAST_STEPS = 10     # Number of steps to forecast



# Importing stock price data using yfinance with given ticker and period
def stock_data(ticker):
    stock_data = yf.Ticker(ticker)
    return stock_data


# Importing stock price data using yfinance with given ticker and period
stock_ticker = stock_data(TICKER)
hist = stock_ticker.history(PERIOD)


# Storing the date index before resetting the index for plotting purposes later
date_index = hist.index 


# Resetting index to integers for predicting purposes
hist = hist.reset_index()



# Identify significant MA lags using ACF
acf_coef = acf(hist['Close'], alpha=0.05)
sig_acf = [
    i for i in range(1, len(acf_coef[0]))
    if (acf_coef[0][i] > (acf_coef[1][i][1] - acf_coef[0][i])) or
    (acf_coef[0][i] < (acf_coef[1][i][0] - acf_coef[0][i]))
]


# Identify significant AR lags using PACF
pacf_coef = pacf(hist['Close'], alpha=0.05)
sig_pacf = [
    i for i in range(1, len(pacf_coef[0]))
    if (pacf_coef[0][i] > (pacf_coef[1][i][1] - pacf_coef[0][i])) or
    (pacf_coef[0][i] < (pacf_coef[1][i][0] - pacf_coef[0][i]))
]



# ARIMA model fitting function
def ARIMA_Price(FORECAST_STEPS):

    # Determine order of integration
    adf_result = adfuller(hist['Close'], autolag = 'BIC')
    d_order = 0 if adf_result[0] < adf_result[4]['5%'] else 1
    
    # ARIMA(p,d,q) Model
    model = ARIMA(
        endog = hist['Close'],
        order = (len(sig_pacf), d_order, len(sig_acf))
    ).fit()
    
    # Generate forecasts
    forecast = model.get_forecast(FORECAST_STEPS, alpha = 0.05)

    # Start predictions after initial period
    initial_period = max(len(sig_pacf), len(sig_acf))  # Use the larger of p or q
    predict = model.predict(start = initial_period)

    # ARIMA model doesn't predict the initial period very well so we have to pad it with initial values
    full_predict = np.full(len(hist['Close']), hist['Close'].iloc[0])
    full_predict[initial_period:] = predict
    
    return forecast.summary_frame(), full_predict



# Fit ARIMA model and generate forecasts
forecast, predict = ARIMA_Price(FORECAST_STEPS)


# Create future dates for forecast
last_date = date_index[-1]
future_dates = pd.date_range(start=last_date, periods=FORECAST_STEPS + 1)[1:]
forecast.index = future_dates


# Calculate error-based metrics
mae = mean_absolute_error(hist['Close'], predict)
mse = mean_squared_error(hist['Close'], predict)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((hist['Close'] - predict) / hist['Close'])) * 100

# Calculate scale-independent metrics
r2 = r2_score(hist['Close'], predict)
adjusted_r2 = 1 - (1-r2) * (len(hist['Close'])-1)/(len(hist['Close'])-predict.shape[0]-1)
smape = 100/len(hist['Close']) * np.sum(2 * np.abs(predict - hist['Close']) / (np.abs(hist['Close']) + np.abs(predict)))

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


# Plot predictions and forecasts
fig = go.Figure()
fig.add_trace(go.Scatter(x = date_index, y = hist['Close'], mode = 'lines', name = 'Observed'))
fig.add_trace(go.Scatter(x = date_index, y = predict, mode = 'lines', name = 'ARIMA Model Prediction'))
fig.add_trace(go.Scatter(x = future_dates, y = forecast['mean'], mode = 'lines', name = 'ARIMA Model Forecast'))

fig.update_layout(
    autosize = False,
    width = 800,
    height = 500,
    xaxis_title = 'Date',
    yaxis_title = 'Price $',
    title = f'{TICKER} Stock Price Forecast'
)

fig.show()