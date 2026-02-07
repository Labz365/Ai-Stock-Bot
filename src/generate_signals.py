import pandas as pd
import numpy as np
import yfinance as yf
import joblib

tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']

# Same drop columns as training — must match exactly
drop_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume',
             'ma_10', 'ma_50', 'volume_ma_10', 'volume_ma_50', 'target']

# Minimum confidence to trigger a BUY signal
BUY_THRESHOLD = 0.60


def build_live_features(ticker):
    """Download recent data and calculate the same features used in training."""
    df = yf.download(ticker, period='120d')
    df.columns = df.columns.get_level_values(0)
    df.reset_index(inplace=True)

    # ==================== RETURNS ====================
    df['return_cc'] = df['Close'].pct_change()
    df['return_oc'] = (df['Close'] - df['Open']) / df['Open']
    df['overnight_gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    df['upside'] = (df['High'] - df['Close'].shift(1)) / df['Close'].shift(1)
    df['downside'] = (df['Low'] - df['Close'].shift(1)) / df['Close'].shift(1)

    # ==================== MOVING AVERAGES ====================
    df['ma_10'] = df['Close'].rolling(window=10).mean()
    df['ma_50'] = df['Close'].rolling(window=50).mean()
    df['price_vs_ma10'] = (df['Close'] - df['ma_10']) / df['ma_10']
    df['price_vs_ma50'] = (df['Close'] - df['ma_50']) / df['ma_50']
    df['ma_diff'] = (df['ma_10'] - df['ma_50']) / df['ma_50']
    df['ma_10_slope'] = df['ma_10'].pct_change(5)
    df['ma_50_slope'] = df['ma_50'].pct_change(5)

    # ==================== VOLATILITY ====================
    df['volatility_10'] = df['return_cc'].rolling(window=10).std()
    df['volatility_50'] = df['return_cc'].rolling(window=50).std()
    df['vol_ratio'] = df['volatility_10'] / df['volatility_50']
    df['volatility_annual'] = df['volatility_10'] * (252 ** 0.5)
    df['parkinson_vol'] = np.sqrt(
        (1 / (4 * np.log(2))) * (np.log(df['High'] / df['Low']) ** 2)
    ).rolling(window=10).mean()
    df['vol_change'] = df['volatility_10'].pct_change(5)

    # ==================== HIGH-LOW FEATURES ====================
    df['intraday_range'] = (df['High'] - df['Low']) / df['Close']
    df['close_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])

    # ==================== RSI (NEW) ====================
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / loss))

    # ==================== MACD (NEW) ====================
    ema_12 = df['Close'].ewm(span=12).mean()
    ema_26 = df['Close'].ewm(span=26).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9).mean()
    macd_hist = macd - macd_signal
    df['macd_norm'] = macd / df['Close']
    df['macd_signal_norm'] = macd_signal / df['Close']
    df['macd_hist_norm'] = macd_hist / df['Close']

    # ==================== VOLUME FEATURES (NEW) ====================
    df['volume_ma_10'] = df['Volume'].rolling(10).mean()
    df['volume_ma_50'] = df['Volume'].rolling(50).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_ma_10']
    df['volume_trend'] = df['volume_ma_10'] / df['volume_ma_50']
    df['price_volume'] = df['return_cc'] * df['volume_ratio']

    # ==================== DAY OF WEEK (NEW) ====================
    date_col = df.columns[0]
    df['day_of_week'] = pd.to_datetime(df[date_col]).dt.dayofweek

    df.dropna(inplace=True)
    return df


def generate_signals():
    """Generate BUY/HOLD signals for all tickers with confidence scores."""
    signals = {}

    for ticker in tickers:
        # Build features from live data
        df = build_live_features(ticker)

        # Get feature columns (same as training)
        feature_cols = [c for c in df.columns if c not in drop_cols]

        # Use only the latest row
        latest = df[feature_cols].iloc[[-1]]

        # Load the trained model
        model = joblib.load(f'models/{ticker}.pkl')

        # Predict
        prediction = model.predict(latest)[0]
        probability = model.predict_proba(latest)[0]
        confidence = probability[1]  # probability of UP

        # Only BUY if model is confident enough
        if prediction == 1 and confidence >= BUY_THRESHOLD:
            signals[ticker] = 'BUY'
        else:
            signals[ticker] = 'HOLD'

        print(f"{ticker}: {signals[ticker]} "
              f"(DOWN: {probability[0]:.2f}, UP: {confidence:.2f}) "
              f"{'✓ above threshold' if confidence >= BUY_THRESHOLD else ''}")

    return signals


if __name__ == '__main__':
    signals = generate_signals()
    print(f"\nFinal signals: {signals}")
