# Stock Price Prediction Using Machine Learning
This project implements a comprehensive stock price forecasting system utilizing multiple advanced machine learning and statistical models. The system provides comparative analysis capabilities across six different prediction approaches, enabling robust stock price forecasting and analysis.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Models and Methodologies](#models-and-methodologies)
  - [ARIMA](#arima)
  - [Transformer](#transformer)
  - [LSTM](#lstm)
  - [XGBoost](#xgboost)
  - [SVR](#support-vector-regression-svr)
  - [Hybrid Model](#hybrid-model)
- [Evaluation Metrics](#evaluation-metrics)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the Models](#running-the-models)

## Overview
Stock price forecasting is critical for informed financial decision-making.Here I have developed a comprehensive stock price forecasting system implementing multiple advanced machine learning and statistical models. The system features six different prediction approaches: ARIMA, Transformer neural networks, LSTM neural networks, XGBoost, Support Vector Regression (SVR), and a novel hybrid model combining Transformer and ARIMA architectures. Created a modular architecture enabling comparative analysis of different prediction methodologies, with each model producing performance metrics (RMSE, MAE, R², MAPE) and interactive visualizations using Plotly. Implemented sophisticated data preprocessing techniques including time series analysis, feature engineering, and data normalization. The hybrid model leverages both statistical and deep learning approaches to enhance prediction accuracy through weighted ensemble predictions.

## Features
- Multiple prediction models:
  - ARIMA (Statistical Time Series Analysis)
  - Transformer Neural Network
  - LSTM Neural Network
  - XGBoost
  - Support Vector Regression (SVR)
  - Hybrid Model (Transformer + ARIMA)
- Interactive visualizations using Plotly
- Comprehensive performance metrics (MAE, MSE, RMSE, R², MAPE)
- Future price forecasting capabilities


## Models and Methodologies

### ARIMA
- **Description**: Statistical model for time series forecasting based on autoregression, differencing, and moving averages.
- **Techniques**: 
  - ADF tests for stationarity
  - ACF/PACF plots for lag selection

### Transformer
- **Description**: A deep learning model inspired by attention mechanisms, designed for sequence-to-sequence tasks. The Transformer model excels in capturing complex relationships in time series data.

### LSTM
- **Description**: Uses a Long Short-Term Memory (LSTM) model to learn temporal patterns and predict sequential data.

### XGBoost
- **Description**: Implements gradient boosting regression to predict stock prices based on engineered features.

### Support Vector Regression (SVR)
- **Description**: A regression model that employs Support Vector Machines, enhanced with technical indicators such as SMA and RSI. Hyperparameters are fine-tuned using Grid Search.

### Hybrid Model
- **Description**: Combines predictions from ARIMA and Transformer using a weighted average approach to improve overall accuracy. This model capitalizes on the strengths of both statistical and machine learning paradigms.

## Evaluation Metrics
Models are evaluated using the following metrics:
- **Mean Squared Error (MSE)**
- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Percentage Error (MAPE)**
- **R-squared (R²)**
- **Adjusted R-squared**
- **Symmetric Mean Absolute Percentage Error (SMAPE)**

## Project Structure
```plaintext
.
├── ARIMA.py          # ARIMA model for statistical forecasting
├── Transformer.py    # Transformer-based neural network for time series forecasting
├── LSTM.py           # LSTM-based sequential model for stock price prediction
├── XGBoost.py        # Gradient boosting regression for stock price forecasting
├── SVR.py            # Support Vector Regression model with feature engineering
├── Hybrid.py         # Hybrid model combining ARIMA and Transformer
└── requirements.txt  # Required Python packages
```

## Installation
To set up the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Ali-Pirhosseinloo/Stock-Price-Prediction-Using-Machine-Learning.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Stock-Price-Prediction-Using-Machine-Learning
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Models

Execute individual scripts to run specific models:
- ARIMA:
  ```bash
  python ARIMA.py
  ```
- Transformer:
  ```bash
  python Transformer.py
  ```
- LSTM:
  ```bash
  python LSTM.py
  ```
- XGBoost:
  ```bash
  python XGBoost.py
  ```
- SVR:
  ```bash
  python SVR.py
  ```
- Hybrid Model:
  ```bash
  python Hybrid.py
  ```
