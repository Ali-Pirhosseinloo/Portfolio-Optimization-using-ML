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
 
  ![ARIMA_Prediction](https://github.com/user-attachments/assets/d1259c5d-e98f-4e9d-9e9d-b180760c85b5)
![ARIMA_Forecast](https://github.com/user-attachments/assets/f08fa2c8-ecb3-4304-a285-1df7396e84be)


```
ARIMA Performance Metrics:
--------------------------------------------------
Error-based metrics:
MAE: 3.7753
MSE: 24.8492
RMSE: 4.9849
MAPE: 1.0907

Scale-independent metrics:
R²: 0.9940
Adjusted R²: 4.0051
SMAPE: 1.0898
```


### Transformer
- **Description**: A deep learning model inspired by attention mechanisms, designed for sequence-to-sequence tasks. The Transformer model excels in capturing complex relationships in time series data.
- **Techniques**:
  - Custom Transformer architecture for time series forecasting
  - Sequence-to-sequence prediction capability

![Transformer_Prediction](https://github.com/user-attachments/assets/f60d858e-9b48-4eb0-a363-fa563be685e4)
![Transformer_Forecast](https://github.com/user-attachments/assets/bff1d1df-f753-4ebe-887b-2c6052d6bcb1)


```
Transformer Performance Metrics:
--------------------------------------------------
Error-based metrics:
MAE: 8.9887
MSE: 108.2051
RMSE: 10.4022
MAPE: 2.6565%

Scale-independent metrics:
R²: 0.9711
Adjusted R²: 0.9710
SMAPE: 2.6073%
```

### LSTM
- **Description**: Uses a Long Short-Term Memory (LSTM) model to learn temporal patterns and predict sequential data.
- **Techniques**: 
  - Rolling window forecasting approach
  - Min-Max scaling for data normalization

![LSTM_Prediction](https://github.com/user-attachments/assets/000f0981-ebf0-4a91-ac1f-1ce7aaf8ed4e)
![LSTM_Forecast](https://github.com/user-attachments/assets/d4d22105-c868-4637-9813-e35a697dcad2)

```
LSTM Performance Metrics:
--------------------------------------------------
Error-based metrics:
MAE: 10.0938
MSE: 131.5272
RMSE: 11.4685
MAPE: 2.3043

Scale-independent metrics:
R²: 0.3085
Adjusted R²: 0.2784
SMAPE: 2.3381
```

### XGBoost
- **Description**: Implements gradient boosting regression to predict stock prices based on engineered features.

![XGBoost_Prediction](https://github.com/user-attachments/assets/c726ffca-9c6e-4d53-902c-fe774626ba8e)
![XGBoost_Forecast](https://github.com/user-attachments/assets/039cbe73-1097-46cb-abe3-ecb5534447d2)

```
XGBoost Performance Metrics:
--------------------------------------------------
Error-based metrics:
MAE: 3.6750
MSE: 21.6540
RMSE: 4.6534
MAPE: 0.8508

Scale-independent metrics:
R²: 0.8875
Adjusted R²: 0.7001
SMAPE: 0.8529
```

### Support Vector Regression (SVR)
- **Description**: A regression model that employs Support Vector Machines, enhanced with technical indicators such as SMA and RSI. Hyperparameters are fine-tuned using Grid Search.

![SVR_Prediction](https://github.com/user-attachments/assets/24fbfb99-d24d-46fd-b34f-1a3f7fb93b35)
![SVR_Forecast](https://github.com/user-attachments/assets/3a363f0e-bf4b-44aa-9fac-de58b68c92d2)

```
SVR Performance Metrics:
--------------------------------------------------
Error-based metrics:
MAE: 3.8170
MSE: 24.3317
RMSE: 4.9327
MAPE: 0.8864

Scale-independent metrics:
R²: 0.8734
Adjusted R²: 1.0844
SMAPE: 0.8867
```

### Hybrid Model
- **Description**: Combines predictions from ARIMA and Transformer using a weighted average approach to improve overall accuracy. This model capitalizes on the strengths of both statistical and machine learning paradigms.

![Hybrid_Prediction](https://github.com/user-attachments/assets/fb9c2771-4119-44e0-bc90-4d9a95f99d29)
![Hybrid_Forecast](https://github.com/user-attachments/assets/c18c84b2-9449-4451-acc3-68132e258ed7)

```
 Transformer Performance Metrics:
--------------------------------------------------
Error-based metrics:
MAE: 6.6409
MSE: 65.1720
RMSE: 8.0729
MAPE: 1.8615

Scale-independent metrics:
R²: 0.9826
Adjusted R²: 0.9825
SMAPE: 1.8442


 Hybrid Model Performance Metrics:
--------------------------------------------------
Error-based metrics:
MAE: 3.9374
MSE: 26.7440
RMSE: 5.1715
MAPE: 1.1079

Scale-independent metrics:
R²: 0.9929
Adjusted R²: 0.9928
SMAPE: 1.1050
```

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
