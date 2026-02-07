import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import api
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import yfinance as yf

tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']

# Trading rules
STOP_LOSS_PCT = -0.02      # -2% stop loss
TAKE_PROFIT_PCT = 0.05     # +5% take profit


# ==================== POSITION SIZING ====================

def get_position_dollars(confidence, account_cash):
    """Calculate dollar amount to invest based on confidence and account size."""
    cash = float(account_cash)

    if confidence >= 0.70:
        return cash * 0.10    # 10% of account
    elif confidence >= 0.65:
        return cash * 0.07    # 7% of account
    elif confidence >= 0.60:
        return cash * 0.05    # 5% of account
    else:
        return 0


def get_shares_from_dollars(ticker, dollars):
    """Convert dollar amount to number of whole shares."""
    try:
        quote = api.get_latest_trade(ticker)
        price = float(quote.price)
        shares = int(dollars // price)
        return max(shares, 0), price
    except Exception as e:
        print(f"Could not get price for {ticker}: {e}")
        return 1, 0.0


# ==================== POSITION & ORDER CHECKS ====================

def get_current_positions():
    positions = {}
    for p in api.list_positions():
        positions[p.symbol] = {
            'qty': int(p.qty),
            'entry_price': float(p.avg_entry_price),
            'current_price': float(p.current_price),
            'pnl_pct': float(p.unrealized_plpc)
        }
    return positions


def get_pending_orders():
    pending = set()
    for order in api.list_orders(status='open'):
        pending.add(order.symbol)
    return pending


def cancel_stale_orders(signals):
    """Cancel pending buy orders where the signal is no longer BUY."""
    for order in api.list_orders(status='open'):
        if order.side == 'buy' and signals.get(order.symbol) != 'BUY':
            api.cancel_order(order.id)
            print(f"Cancelled stale BUY order for {order.symbol}")


# ==================== CONFIDENCE CALCULATOR ====================

def get_confidences():
    """Get the latest prediction confidences from the models."""
    drop_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume',
                 'ma_10', 'ma_50', 'volume_ma_10', 'volume_ma_50', 'target']
    confidences = {}

    for ticker in tickers:
        try:
            model = joblib.load(f'models/{ticker}.pkl')
            df = yf.download(ticker, period='120d', progress=False)
            df.columns = df.columns.get_level_values(0)
            df.reset_index(inplace=True)

            # Feature calculation
            df['return_cc'] = df['Close'].pct_change()
            df['return_oc'] = (df['Close'] - df['Open']) / df['Open']
            df['overnight_gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
            df['upside'] = (df['High'] - df['Close'].shift(1)) / df['Close'].shift(1)
            df['downside'] = (df['Low'] - df['Close'].shift(1)) / df['Close'].shift(1)
            df['ma_10'] = df['Close'].rolling(10).mean()
            df['ma_50'] = df['Close'].rolling(50).mean()
            df['price_vs_ma10'] = (df['Close'] - df['ma_10']) / df['ma_10']
            df['price_vs_ma50'] = (df['Close'] - df['ma_50']) / df['ma_50']
            df['ma_diff'] = (df['ma_10'] - df['ma_50']) / df['ma_50']
            df['ma_10_slope'] = df['ma_10'].pct_change(5)
            df['ma_50_slope'] = df['ma_50'].pct_change(5)
            df['volatility_10'] = df['return_cc'].rolling(10).std()
            df['volatility_50'] = df['return_cc'].rolling(50).std()
            df['vol_ratio'] = df['volatility_10'] / df['volatility_50']
            df['volatility_annual'] = df['volatility_10'] * (252 ** 0.5)
            df['parkinson_vol'] = np.sqrt(
                (1 / (4 * np.log(2))) * (np.log(df['High'] / df['Low']) ** 2)
            ).rolling(10).mean()
            df['vol_change'] = df['volatility_10'].pct_change(5)
            df['intraday_range'] = (df['High'] - df['Low']) / df['Close']
            df['close_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            df['rsi'] = 100 - (100 / (1 + gain / loss))
            ema_12 = df['Close'].ewm(span=12).mean()
            ema_26 = df['Close'].ewm(span=26).mean()
            macd = ema_12 - ema_26
            macd_signal = macd.ewm(span=9).mean()
            df['macd_norm'] = macd / df['Close']
            df['macd_signal_norm'] = macd_signal / df['Close']
            df['macd_hist_norm'] = (macd - macd_signal) / df['Close']
            df['volume_ma_10'] = df['Volume'].rolling(10).mean()
            df['volume_ma_50'] = df['Volume'].rolling(50).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_ma_10']
            df['volume_trend'] = df['volume_ma_10'] / df['volume_ma_50']
            df['price_volume'] = df['return_cc'] * df['volume_ratio']
            date_col = df.columns[0]
            df['day_of_week'] = pd.to_datetime(df[date_col]).dt.dayofweek

            df.dropna(inplace=True)
            feature_cols = [c for c in df.columns if c not in drop_cols]
            latest = df[feature_cols].iloc[[-1]]
            probability = model.predict_proba(latest)[0]
            confidences[ticker] = probability[1]
        except Exception as e:
            print(f"Could not get confidence for {ticker}: {e}")
            confidences[ticker] = 0.5

    return confidences


# ==================== MAIN TRADE EXECUTION ====================

def execute_trades(signals):
    cancel_stale_orders(signals)
    positions = get_current_positions()
    pending = get_pending_orders()
    confidences = get_confidences()
    account = api.get_account()
    log = []

    print(f"\n=== Trade Execution {datetime.now().strftime('%Y-%m-%d %H:%M')} ===")
    print(f"Cash available: ${float(account.cash):,.2f}")
    print(f"Portfolio value: ${float(account.portfolio_value):,.2f}")
    print(f"Current positions: {list(positions.keys()) if positions else 'None'}")
    print(f"Pending orders: {list(pending) if pending else 'None'}")
    print(f"Signals: {signals}\n")

    for ticker in tickers:
        signal = signals.get(ticker, 'HOLD')
        has_position = ticker in positions
        confidence = confidences.get(ticker, 0.5)
        action = 'NONE'

        if has_position:
            pnl = positions[ticker]['pnl_pct']
            qty = positions[ticker]['qty']

            if pnl <= STOP_LOSS_PCT:
                # Stop-loss triggered
                api.submit_order(symbol=ticker, qty=qty, side='sell',
                                 type='market', time_in_force='gtc')
                action = f'SELL {qty} shares (stop-loss hit: {pnl:.2%})'

            elif pnl >= TAKE_PROFIT_PCT:
                # Take profit triggered
                api.submit_order(symbol=ticker, qty=qty, side='sell',
                                 type='market', time_in_force='gtc')
                action = f'SELL {qty} shares (take-profit: {pnl:.2%})'

            elif signal == 'HOLD':
                # Model says sell
                api.submit_order(symbol=ticker, qty=qty, side='sell',
                                 type='market', time_in_force='gtc')
                action = f'SELL {qty} shares (signal)'

            else:
                action = f'HOLD (keeping {qty} shares, P/L: {pnl:.2%})'

        elif signal == 'BUY':
            if ticker in pending:
                action = 'SKIP (order already pending)'
            else:
                # Refresh account cash (it changes as we place orders)
                account = api.get_account()
                dollars = get_position_dollars(confidence, account.cash)
                shares, price = get_shares_from_dollars(ticker, dollars)

                if shares > 0:
                    api.submit_order(symbol=ticker, qty=shares, side='buy',
                                     type='market', time_in_force='gtc')
                    action = (f'BUY {shares} shares @ ~${price:.2f} '
                              f'(~${dollars:,.0f}, {confidence:.0%} confidence)')
                else:
                    action = f'SKIP (not enough cash for 1 share, need ~${price:.2f})'
        else:
            action = 'SKIP (no position, no buy signal)'

        print(f"{ticker}: {action}")
        log.append({
            'timestamp': datetime.now().isoformat(),
            'ticker': ticker,
            'signal': signal,
            'action': action,
            'confidence': round(confidence, 4)
        })

    return log


if __name__ == '__main__':
    test_signals = {
        'AAPL': 'BUY',
        'MSFT': 'HOLD',
        'GOOGL': 'BUY',
        'AMZN': 'HOLD',
        'NVDA': 'BUY'
    }
    execute_trades(test_signals)
