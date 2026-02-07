# Ai-Stock-Bot
Creating an ai stock bot with the help of claude
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

There might be some conflict between the module websockets in both the alpaca trade api and the yfinance api the order this is installed in affects the program.
