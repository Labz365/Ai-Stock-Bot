

import pandas as pd
import numpy as np

tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']

for ticker in tickers:
    df = pd.read_csv(f'data/{ticker}.csv', parse_dates=[0])

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
    df['ema_12'] = df['Close'].ewm(span=12).mean()
    df['ema_26'] = df['Close'].ewm(span=26).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    # Normalize MACD by price so it's comparable across stocks
    df['macd_norm'] = df['macd'] / df['Close']
    df['macd_signal_norm'] = df['macd_signal'] / df['Close']
    df['macd_hist_norm'] = df['macd_hist'] / df['Close']

    # ==================== VOLUME FEATURES (NEW) ====================
    df['volume_ma_10'] = df['Volume'].rolling(10).mean()
    df['volume_ma_50'] = df['Volume'].rolling(50).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_ma_10']
    df['volume_trend'] = df['volume_ma_10'] / df['volume_ma_50']
    # Price-volume relationship
    df['price_volume'] = df['return_cc'] * df['volume_ratio']

    # ==================== DAY OF WEEK (NEW) ====================
    date_col = df.columns[0]
    df['day_of_week'] = pd.to_datetime(df[date_col]).dt.dayofweek

    # ==================== TARGET ====================
    threshold = 0.002
    df['target'] = (df['Close'].pct_change().shift(-1) > threshold).astype(int)

    # ==================== CLEANUP ====================
    # Drop raw MACD/EMA columns (keep normalized versions)
    df.drop(columns=['ema_12', 'ema_26', 'macd', 'macd_signal', 'macd_hist'], inplace=True)

    print(f"{ticker}: {len(df)} rows before cleanup")
    print(df.isna().sum()[df.isna().sum() > 0])
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"{ticker}: {len(df)} rows after cleanup")

    # Save
    df.to_csv(f'data/{ticker}_features.csv', index=False)
    print(f"Saved data/{ticker}_features.csv\n")
