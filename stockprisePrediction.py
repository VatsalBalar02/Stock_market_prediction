import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import TimeSeriesSplit

# Download stock data
symbol = 'AAPL'
start_date = '2010-01-01'  # Extended historical data
end_date = '2024-01-01'
stock_data = yf.download(symbol, start=start_date, end=end_date)

# Technical Indicators
def calculate_technical_indicators(data):
    data['SMA20'] = data['Close'].rolling(window=20).mean()
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    data['RSI14'] = calculate_RSI(data['Close'], 14)
    data['MACD'], data['Signal'] = calculate_MACD(data)
    data['BB_upper'], data['BB_lower'] = calculate_bollinger_bands(data['Close'])
    data['ADX'] = calculate_ADX(data)
    data['OBV'] = calculate_OBV(data)
    data['ATR'] = calculate_ATR(data)
    data['Price_Lag1'] = data['Close'].shift(1)
    data['Price_Lag2'] = data['Close'].shift(2)
    data['Returns'] = data['Close'].pct_change()
    data['Volatility'] = data['Returns'].rolling(window=20).std()
    data.dropna(inplace=True)
    return data

def calculate_RSI(close, window=14):
    delta = close.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    RS = gain / loss
    return 100 - (100 / (1 + RS))

def calculate_MACD(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

def calculate_bollinger_bands(close, window=20, num_std=2):
    rolling_mean = close.rolling(window=window).mean()
    rolling_std = close.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

def calculate_ADX(data, period=14):
    high = data['High']
    low = data['Low']
    close = data['Close']
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / atr)
    minus_di = abs(100 * (minus_dm.ewm(alpha=1/period).mean() / atr))
    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    adx = ((dx.shift(1) * (period - 1)) + dx) / period
    adx_smooth = adx.ewm(alpha=1/period).mean()
    return adx_smooth

def calculate_OBV(data):
    obv = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()
    return obv

def calculate_ATR(data, period=14):
    high = data['High']
    low = data['Low']
    close = data['Close']
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
    atr = tr.rolling(period).mean()
    return atr

# Prepare data with technical indicators
stock_data = calculate_technical_indicators(stock_data)

# Select features for the model
features = ['Close', 'SMA20', 'SMA50', 'RSI14', 'MACD', 'Signal', 'BB_upper', 'BB_lower', 'ADX', 'OBV', 'ATR', 'Returns', 'Volatility']

# Scaling data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(stock_data[features])

# Prepare the dataset for LSTM
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i])
        y.append(data[i, 0])  # Predict 'Close' price
    return np.array(X), np.array(y)

# Create train and test sets
time_step = 60  # Number of past days to use for predicting
X, y = create_dataset(scaled_data, time_step)

# Use TimeSeriesSplit for validation
tscv = TimeSeriesSplit(n_splits=5)

# Build LSTM model
def build_model(input_shape):
    model = Sequential([
        Bidirectional(LSTM(100, return_sequences=True), input_shape=input_shape),
        Dropout(0.2),
        Bidirectional(LSTM(100, return_sequences=True)),
        Dropout(0.2),
        Bidirectional(LSTM(50)),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# Train the model using cross-validation
cv_scores = []
predictions = []
for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    model = build_model((X_train.shape[1], X_train.shape[2]))
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
    
    history = model.fit(X_train, y_train, 
                        epochs=100, 
                        batch_size=32, 
                        validation_data=(X_test, y_test),
                        callbacks=[early_stopping, reduce_lr],
                        verbose=0)
    
    score = model.evaluate(X_test, y_test, verbose=0)
    cv_scores.append(score)
    
    pred = model.predict(X_test)
    predictions.extend(pred.flatten())

print(f"Cross-validation MSE: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

# Prepare the final predictions
final_predictions = scaler.inverse_transform(np.column_stack((predictions, np.zeros((len(predictions), len(features)-1)))))[:, 0]

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(stock_data.index[-len(final_predictions):], stock_data['Close'][-len(final_predictions):], label='Actual Prices')
plt.plot(stock_data.index[-len(final_predictions):], final_predictions, label='LSTM Predictions', color='red', linestyle='--')
plt.title('Actual vs Predicted Stock Prices (LSTM with Cross-validation)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()


# Predict future stock prices
future_days = 30
last_sequence = scaled_data[-time_step:].reshape(1, time_step, len(features))
future_predictions = []

for _ in range(future_days):
    future_price = model.predict(last_sequence)
    future_predictions.append(future_price[0, 0])
    
    # Prepare the next sequence
    last_sequence = np.roll(last_sequence, -1, axis=1)
    last_sequence[0, -1, 0] = future_price[0, 0]  # Update only the 'Close' price
    
    
# Inverse transform future predictions
future_predictions = np.array(future_predictions).reshape(-1, 1)
future_predictions = scaler.inverse_transform(np.column_stack((future_predictions, np.zeros((len(future_predictions), len(features)-1)))))[:, 0]

# Plot future predictions
plt.figure(figsize=(12, 6))
future_dates = pd.date_range(start=stock_data.index[-1] + pd.Timedelta(days=1), periods=future_days)
plt.plot(future_dates, future_predictions, label='Future Price Predictions', color='green')
plt.title('Future Stock Price Predictions (LSTM)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

# Print the predicted prices
for date, price in zip(future_dates, future_predictions):
    print(f"{date.date()}: ${price:.2f}")
