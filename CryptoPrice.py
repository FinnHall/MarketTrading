from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoLatestTradeRequest

# No keys needed for crypto market data
client = CryptoHistoricalDataClient()

# Get the latest trade for ETH/USD
trade = client.get_crypto_latest_trade(CryptoLatestTradeRequest(symbol_or_symbols="ETH/USD"))

# Access the price
print(f'ETH/USD latest price: ${trade["ETH/USD"].price}')