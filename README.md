# Stock Price Prediction using LSTM

This project implements a stock price prediction model using Long Short-Term Memory (LSTM) neural networks. It uses historical stock data and various technical indicators to forecast future stock prices.

## Features

- Downloads historical stock data using yfinance
- Calculates multiple technical indicators:
  - Simple Moving Averages (SMA)
  - Relative Strength Index (RSI)
  - Moving Average Convergence Divergence (MACD)
  - Bollinger Bands
  - Average Directional Index (ADX)
  - On-Balance Volume (OBV)
  - Average True Range (ATR)
- Implements a Bidirectional LSTM model for prediction
- Uses Time Series Cross-Validation for robust model evaluation
- Predicts future stock prices for the next 30 days

## Requirements

- Python 3.7+
- yfinance
- pandas
- numpy
- matplotlib
- scikit-learn
- tensorflow

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/your-username/stock-price-prediction.git
   cd stock-price-prediction
   ```
