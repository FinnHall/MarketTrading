import os
import pandas as pd
import matplotlib.pyplot as plt
import logging
from datetime import datetime, timedelta, timezone

from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Retrieve API keys from environment variables
API_KEY = os.getenv('ALPACA_API_KEY')
SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
if not API_KEY or not SECRET_KEY:
    raise Exception('API keys not set in environment variables.')

# Create an Alpaca historical data client
data_client = CryptoHistoricalDataClient(API_KEY, SECRET_KEY)

# Define the time window for historical data (last 30 days)
end_time = datetime.now(timezone.utc)
start_time = end_time - timedelta(days=30)

# Prepare the request parameters
request_params = CryptoBarsRequest(
    symbol_or_symbols=['ETH/USD'],
    timeframe=TimeFrame.Hour,
    start=start_time,
    end=end_time
)

# Fetch the historical bars data from Alpaca
bars = data_client.get_crypto_bars(request_params).df

# Filter the DataFrame for 'ETH/USD' and adjust the index if necessary
if 'symbol' in bars.index.names:
    df = bars[bars.index.get_level_values('symbol') == 'ETH/USD']
    df = df.reset_index(level='symbol', drop=True)
else:
    df = bars

# Print the first few rows for verification
logging.info('Historical data sample:')
logging.info(df.head())

# Calculate a Simple Moving Average (SMA) for additional filtering
sma_period = 20
if 'close' in df.columns:
    df['sma'] = df['close'].rolling(window=sma_period).mean()
else:
    raise Exception('DataFrame does not contain a close column.')

# Calculate Relative Strength Index (RSI)
rsi_period = 14

delta = df['close'].diff()
# Compute gains and losses
gain = delta.copy()
loss = delta.copy()

gain[gain < 0] = 0
loss[loss > 0] = 0
loss = loss.abs()

avg_gain = gain.rolling(window=rsi_period).mean()
avg_loss = loss.rolling(window=rsi_period).mean()

rs = avg_gain / avg_loss

df['rsi'] = 100 - (100 / (1 + rs))

# Calculate Average True Range (ATR)
atr_period = 14

# Compute True Range components
df['previous_close'] = df['close'].shift(1)
df['high_low'] = df['high'] - df['low']
df['high_pc'] = (df['high'] - df['previous_close']).abs()
df['low_pc'] = (df['low'] - df['previous_close']).abs()

# True Range is the max of the three components
df['tr'] = df[['high_low', 'high_pc', 'low_pc']].max(axis=1)

# ATR is the rolling average of True Range
df['atr'] = df['tr'].rolling(window=atr_period).mean()

# Strategy parameters
initial_capital = 10000  # Starting capital in USD
position = 0             # Current ETH holdings (in units of ETH)
capital = initial_capital
entry_price = None
max_price = None

# Define your strategy thresholds
buy_threshold = 5000      # Example: Buy when price dips below this threshold
atr_multiplier_stop = 2.0  # Initial stop loss multiplier (e.g., 1.5 * ATR)
atr_multiplier_trail = 1.5  # Trailing stop multiplier (e.g., 1.0 * ATR)

# Transaction fee rate (e.g., 0.1% per trade)
fee_rate = 0.002

trade_log = []
# To record portfolio equity over time
equity_curve = []

# Backtesting loop: iterate over the historical data
for current_time, row in df.iterrows():
    price = row['close']
    # Only proceed if SMA is available (skip initial periods)
    if pd.isna(row['sma']):
        continue

    # Record current equity: if in position, equity = position * price, else capital
    current_equity = position * price if position > 0 else capital
    equity_curve.append({'time': current_time, 'equity': current_equity})

    # Buy condition: if not in a position, the price is below the buy threshold, and price is below its SMA
    if position == 0 and price < buy_threshold and price < row['sma'] and row['rsi'] < 25:
        # Apply transaction fee when buying
        effective_capital = capital * (1 - fee_rate)
        position = effective_capital / price  # Buy as many ETH as possible
        capital = 0
        entry_price = price
        max_price = price
        trade_log.append({'action': 'buy', 'price': price, 'time': current_time})
        logging.info(f"{current_time}: BUY at {price:.2f}")

    # Sell condition: if holding a position, check for ATR-based stop loss or trailing stop
    if position > 0:
        current_atr = row['atr']
        # Compute initial stop level using ATR
        initial_stop = entry_price - (atr_multiplier_stop * current_atr)

        # Update the trailing maximum price only if the current price exceeds the entry price
        if price > entry_price:
            max_price = max(max_price, price)

        # Compute trailing stop level if a clear peak exists
        trailing_stop_level = None
        if max_price is not None and max_price > entry_price:
            trailing_stop_level = max_price - (atr_multiplier_trail * current_atr)

        # Sell if price falls below the initial stop or (if available) the trailing stop level
        if price <= initial_stop or (trailing_stop_level is not None and price <= trailing_stop_level):
            effective_sale = position * price * (1 - fee_rate)
            capital = effective_sale
            trade_log.append({'action': 'sell', 'price': price, 'time': current_time})
            logging.info(f"{current_time}: SELL at {price:.2f} (Initial Stop: {initial_stop:.2f}, Trailing Stop: {trailing_stop_level if trailing_stop_level is not None else 'N/A'})")
            position = 0
            entry_price = None
            max_price = None

# Finalize: if still holding a position, sell at the last available price
if position > 0:
    final_price = df.iloc[-1]['close']
    effective_sale = position * final_price * (1 - fee_rate)
    capital = effective_sale
    trade_log.append({'action': 'sell', 'price': final_price, 'time': df.index[-1]})
    logging.info(f"{df.index[-1]}: Final SELL at {final_price:.2f}")
    position = 0

# Record final equity
if equity_curve:
    equity_curve.append({'time': df.index[-1], 'equity': capital})

logging.info(f"Final capital: ${capital:.2f}")

# Compute performance metrics
total_return = (capital - initial_capital) / initial_capital * 100
num_trades = len([trade for trade in trade_log if trade['action'] == 'sell'])
logging.info(f"Total Return: {total_return:.2f}%")
logging.info(f"Total Trades Executed: {num_trades}")

# Compute trading duration in days using the first and last dates in the DataFrame
trading_duration = (df.index[-1] - df.index[0]).days
logging.info(f"Trading Duration: {trading_duration} days")

# Plot the historical price data and mark the buy/sell signals
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['close'], label='Close Price')

# Separate the buy and sell signals from the trade log
buy_signals = [trade for trade in trade_log if trade['action'] == 'buy']
sell_signals = [trade for trade in trade_log if trade['action'] == 'sell']

plt.scatter([trade['time'] for trade in buy_signals], [trade['price'] for trade in buy_signals],
            marker='^', color='g', label='Buy', s=100)
plt.scatter([trade['time'] for trade in sell_signals], [trade['price'] for trade in sell_signals],
            marker='v', color='r', label='Sell', s=100)

plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Backtesting Results with SMA Filter and Fees')
plt.legend()
plt.show()

# Plot the equity curve
if equity_curve:
    equity_df = pd.DataFrame(equity_curve)
    equity_df.set_index('time', inplace=True)
    plt.figure(figsize=(10, 5))
    plt.plot(equity_df.index, equity_df['equity'], label='Equity Curve')
    plt.xlabel('Time')
    plt.ylabel('Equity ($)')
    plt.title('Portfolio Equity Curve')
    plt.legend()
    plt.show()