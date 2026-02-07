import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd
from config import api
from datetime import datetime

LOG_FILE = 'data/trade_log.json'

def load_log():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            return json.load(f)
    return []

def monitor():
    print("=" * 50)
    print(f"PERFORMANCE REPORT: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 50)

    # --- Account Summary ---
    account = api.get_account()
    starting_cash = 100000.0
    current_value = float(account.portfolio_value)
    total_return = ((current_value - starting_cash) / starting_cash) * 100

    print(f"\nCash: ${account.cash}")
    print(f"Portfolio value: ${current_value:,.2f}")
    print(f"Total return: {total_return:+.2f}%")

    # --- Current Positions ---
    positions = api.list_positions()
    if positions:
        print(f"\n--- Open Positions ({len(positions)}) ---")
        for p in positions:
            pnl = float(p.unrealized_pl)
            pnl_pct = float(p.unrealized_plpc) * 100
            print(f"{p.symbol}: {p.qty} shares | "
                  f"Entry: ${float(p.avg_entry_price):.2f} | "
                  f"Current: ${float(p.current_price):.2f} | "
                  f"P/L: ${pnl:+.2f} ({pnl_pct:+.2f}%)")
    else:
        print("\nNo open positions")

    # --- Trade History from Log ---
    log = load_log()
    if not log:
        print("\nNo trade history yet")
        return

    print(f"\n--- Trade History ({len(log)} bot runs) ---")

    total_buys = 0
    total_sells = 0
    total_skips = 0

    for entry in log:
        for trade in entry.get('trades', []):
            action = trade.get('action', '')
            if action.startswith('BUY'):
                total_buys += 1
            elif action.startswith('SELL'):
                total_sells += 1
            elif action.startswith('SKIP'):
                total_skips += 1

    print(f"Total BUYs: {total_buys}")
    print(f"Total SELLs: {total_sells}")
    print(f"Total SKIPs: {total_skips}")

    # --- Portfolio Value Over Time ---
    values = []
    for entry in log:
        values.append({
            'date': entry['run_time'][:10],
            'value': float(entry.get('portfolio_value', 0))
        })

    if len(values) >= 2:
        df = pd.DataFrame(values)
        peak = df['value'].max()
        trough = df['value'].min()
        drawdown = ((trough - peak) / peak) * 100

        print(f"\n--- Risk Metrics ---")
        print(f"Peak value: ${peak:,.2f}")
        print(f"Lowest value: ${trough:,.2f}")
        print(f"Max drawdown: {drawdown:.2f}%")
        print(f"First run: {values[0]['date']}")
        print(f"Latest run: {values[-1]['date']}")

    # --- Closed Orders from Alpaca ---
    orders = api.list_orders(status='closed', limit=20)
    if orders:
        print(f"\n--- Recent Closed Orders (last 20) ---")
        for o in orders:
            price = f"${float(o.filled_avg_price):.2f}" if o.filled_avg_price else "pending"
            print(f"{str(o.submitted_at)[:10]} | {o.side.upper()} {o.symbol} | "
                  f"{o.qty} shares @ {price} | "
                  f"Status: {o.status}")

    # --- Equity Curve Chart ---
    if len(values) >= 2:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import yfinance as yf

        df = pd.DataFrame(values)
        df['date'] = pd.to_datetime(df['date'])

        spy = yf.download('SPY', start=df['date'].iloc[0].strftime('%Y-%m-%d'),
                          end=datetime.now().strftime('%Y-%m-%d'))
        spy.columns = spy.columns.get_level_values(0)
        spy_start = spy['Close'].iloc[0]
        spy_end = spy['Close'].iloc[-1]
        spy_return = ((spy_end - spy_start) / spy_start) * 100

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(df['date'], df['value'], color='#2196F3', linewidth=2,
                label=f'Bot ({total_return:+.2f}%)')
        ax.axhline(y=starting_cash, color='gray', linestyle='--',
                    alpha=0.5, label='Starting Cash')

        ax.set_title('Portfolio Value Over Time', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        plt.xticks(rotation=45)
        plt.tight_layout()

        chart_path = 'data/equity_curve.png'
        plt.savefig(chart_path, dpi=150)
        print(f"\nEquity curve saved to {chart_path}")
        plt.show()

        print(f"\n--- Comparison ---")
        print(f"Bot return: {total_return:+.2f}%")
        print(f"SPY buy-and-hold return: {float(spy_return):+.2f}%")
        if total_return > float(spy_return):
            print(">>> Bot is BEATING the market")
        else:
            print(">>> Bot is UNDERPERFORMING the market")

    print("\n" + "=" * 50)
    print("END OF REPORT")

if __name__ == '__main__':
    monitor()
