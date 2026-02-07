import yfinance as yf
import pandas as pd
from datetime import datetime

tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']

for ticker in tickers:
    print(f"Downloading data for {ticker}...")
    data = yf.download(ticker, start="2018-01-01", end=datetime.now().strftime('%Y-%m-%d'))
    data.columns = data.columns.get_level_values(0) 
    data.to_csv(f'data/{ticker}.csv')
    print(f"Data for {ticker} saved to data/{ticker}.csv")

print("All data downloaded and saved successfully.")
