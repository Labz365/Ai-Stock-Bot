import sys
import os
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))

from generate_signals import generate_signals
from execute_trades import execute_trades
from config import api

LOG_FILE = 'data/trade_log.json'

def load_log():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            return json.load(f)
    return []

def save_log(log):
    with open(LOG_FILE, 'w') as f:
        json.dump(log, f, indent=2)

def run():
    if datetime.now().weekday() >= 5:
        print("Weekend — skipping")
        return

    print("=" * 50)
    print(f"BOT RUN: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 50)

    account = api.get_account()
    print(f"\nCash: ${account.cash}")
    print(f"Portfolio value: ${account.portfolio_value}")

    print("\n--- Generating Signals ---")
    signals = generate_signals()

    print("\n--- Executing Trades ---")
    trade_log = execute_trades(signals)

    full_log = load_log()
    full_log.append({
        'run_time': datetime.now().isoformat(),
        'cash': account.cash,
        'portfolio_value': account.portfolio_value,
        'signals': signals,
        'trades': trade_log
    })
    save_log(full_log)

    print(f"\nLog saved to {LOG_FILE}")
    print("=" * 50)
    print("DONE")

if __name__ == '__main__':
    run()
