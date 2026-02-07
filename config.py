from alpaca_trade_api import REST

API_KEY = 'your-api-key-here'
SECRET_KEY = 'your-secret-key-here'
BASE_URL = 'https://paper-api.alpaca.markets'

api = REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')

if __name__ == '__main__':
    account = api.get_account()
    print(f"Account status: {account.status}")
    print(f"Cash: ${account.cash}")
    print(f"Portfolio value: ${account.portfolio_value}")
