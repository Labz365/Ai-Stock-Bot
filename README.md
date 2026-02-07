# AI Stock Trading Bot

An automated paper-trading system that uses machine learning to trade the top 5 most liquid US stocks.

## Stocks Traded
- AAPL (Apple)
- MSFT (Microsoft)
- GOOGL (Alphabet)
- AMZN (Amazon)
- NVDA (Nvidia)

## Features
- Random Forest & Gradient Boosting models (picks the best per stock)
- 20+ technical indicators (RSI, MACD, volatility, volume, moving averages)
- Automated daily signal generation (BUY/HOLD)
- Smart position sizing based on model confidence and account balance
- Stop-loss (-2%) and take-profit (+5%) rules
- Stale order cancellation
- Trade logging and performance monitoring
- Equity curve charting with S&P 500 comparison

## Project Structure
```
AI Stock bot/
├── data/                    # CSV data, feature files, logs
├── models/                  # Trained model files (.pkl)
├── src/
│   ├── download_data.py     # Download OHLCV data from Yahoo Finance
│   ├── build_features.py    # Calculate technical indicators
│   ├── train_model.py       # Train ML models per stock
│   ├── generate_signals.py  # Generate daily BUY/HOLD signals
│   ├── execute_trades.py    # Execute trades via Alpaca API
│   ├── run_bot.py           # Main bot runner (headless)
│   ├── run_bot_gui.py       # Bot runner with GUI
│   ├── retrain.py           # Monthly model refresh script
│   └── monitor_performance.py  # Performance reporting
├── config.py                # API keys (not tracked in git)
├── config_example.py        # Template for config.py
├── run_bot.bat              # Windows Task Scheduler trigger
├── requirements.txt
└── README.md
```

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/ai-stock-bot.git
cd ai-stock-bot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up Alpaca API
- Sign up at [https://alpaca.markets](https://alpaca.markets)
- Enable Paper Trading
- Copy your API Key and Secret Key
- Rename `config_example.py` to `config.py`
- Paste your keys into `config.py`

### 4. Download data and train models
```bash
python src/download_data.py
python src/build_features.py
python src/train_model.py
```

### 5. Run the bot
```bash
python src/run_bot.py
```

Or with the GUI:
```bash
python src/run_bot_gui.py
```

### 6. Monitor performance
```bash
python src/monitor_performance.py
```

### 7. Retrain models (monthly)
```bash
python src/retrain.py
```

## Trading Rules
- Max 1 position per stock
- Position size: 5-10% of account based on model confidence
- No leverage, no shorting
- Stop-loss: -2%
- Take-profit: +5%
- Trade once per day
- Minimum 60% model confidence to trigger a BUY

## Technical Indicators Used
- Daily returns (close-to-close, open-to-close, overnight gap)
- Moving averages (10-day, 50-day) with crossovers and slopes
- Volatility (rolling std, Parkinson, annual, ratio)
- RSI (14-day)
- MACD (normalized)
- Volume ratio and trend
- Intraday range and close position
- Day of week

## Disclaimer
This is a paper-trading educational project. Past performance does not guarantee future results. Do not use real money without thorough testing. This is not financial advice.
