import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import time
from datetime import datetime, timedelta
import pytz
import torch
import torch.nn as nn
import torch.nn.functional as F

# Initialize MT5 connection
if not mt5.initialize():
    print("initialize() failed")
    mt5.shutdown()

# Set symbol and timeframe
symbol = 'BTCUSD'
timeframe = mt5.TIMEFRAME_M1

# Load the trained model
# Define the model architecture
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 469)
        self.fc2 = nn.Linear(469, 290)
        self.output = nn.Linear(290, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)
        return x

STATE_SIZE = 26 * 11 + 1  # Adjusted based on your code
ACTION_SIZE = 4

# Create model instance
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DQN(STATE_SIZE, ACTION_SIZE).to(device)

# Load trained weights
model.load_state_dict(torch.load('dqn_trading_model_best_marge_2-1.pth', map_location=device))
model.eval()  # Set model to evaluation mode

# Initialize variables for positions
position_buy = None  # 'buy' or None
position_sell = None  # 'sell' or None
entry_price_buy = 0.0
entry_price_sell = 0.0
wait_buy = 0
wait_sell = 0
previous_action = 0
profit_buy = 0.0
profit_sell = 0.0
drawdown = 0.0
drawup = 0.0
previous_profit_buy = 0.0
previous_profit_sell = 0.0

def update_positions():
    global position_buy, position_sell, entry_price_buy, entry_price_sell
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        print("Failed to get positions")
        position_buy = None
        position_sell = None
        entry_price_buy = 0.0
        entry_price_sell = 0.0
    else:
        # Reset positions
        position_buy = None
        position_sell = None
        entry_price_buy = 0.0
        entry_price_sell = 0.0
        for pos in positions:
            if pos.type == mt5.POSITION_TYPE_BUY:
                position_buy = 'buy'
                entry_price_buy = pos.price_open
                print(f"Detected open BUY position at price {entry_price_buy}")
            elif pos.type == mt5.POSITION_TYPE_SELL:
                position_sell = 'sell'
                entry_price_sell = pos.price_open
                print(f"Detected open SELL position at price {entry_price_sell}")

def get_account_balance():
    account_info = mt5.account_info()
    if account_info is not None:
        return account_info.balance
    else:
        print("Failed to get account info")
        return None

def get_state():
    # Get the time of the last closed bar
    utc_from = datetime.now(pytz.utc) - timedelta(minutes=15*150)  # Adjust for M15 timeframe and enough data
    rates = mt5.copy_rates_from(symbol, timeframe, utc_from, 200)
    if rates is None or len(rates) < 100:
        print("Not enough data retrieved")
        return None
    # Convert to pandas DataFrame
    data = pd.DataFrame(rates)
    data['time'] = pd.to_datetime(data['time'], unit='s')
    # Now data has columns: time, open, high, low, close, tick_volume, spread, real_volume
    # We need to calculate Ichimoku indicators
    # Tenkan-sen (Conversion Line)
    high_prices = data['high']
    low_prices = data['low']
    close_prices = data['close']
    data['tenkan_sen'] = (high_prices.rolling(window=9).max() + low_prices.rolling(window=9).min()) / 2
    # Kijun-sen (Base Line)
    data['kijun_sen'] = (high_prices.rolling(window=26).max() + low_prices.rolling(window=26).min()) / 2
    # Senkou Span A (Leading Span A)
    data['senkou_span_a'] = ((data['tenkan_sen'] + data['kijun_sen']) / 2).shift(26)
    # Senkou Span B (Leading Span B)
    data['senkou_span_b'] = ((high_prices.rolling(window=52).max() + low_prices.rolling(window=52).min()) / 2).shift(26)
    # Chikou Span (Lagging Span)
    data['chikou_span'] = close_prices.shift(-26)
    # Drop rows with NaN values
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)
    if len(data) < 26:
        print("Not enough data after dropping NaNs")
        return None
    # Now prepare the state as per the training code
    # Take the last 26 periods
    state = []
    data_len = len(data)
    for i in range(data_len - 26, data_len):
        data_i = data.iloc[i]
        # For past senkou_span_a and senkou_span_b (26 periods in the past)
        if i - 26 >= 0:
            past_senkou_span_a = data.iloc[i - 26]['senkou_span_a']
            past_senkou_span_b = data.iloc[i - 26]['senkou_span_b']
        else:
            past_senkou_span_a = 0.0
            past_senkou_span_b = 0.0
        state_i = np.concatenate([
            data_i[['open', 'high', 'low', 'close', 'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span']].astype(float).values,
            [float(past_senkou_span_a), float(past_senkou_span_b)]
        ])
        state.append(state_i)
    state = np.array(state).flatten()
    # Append profit_sell, profit_buy, order_state
    # profit_sell, profit_buy, position_buy, position_sell need to be maintained
    global profit_sell, profit_buy, position_buy, position_sell
    order_state = 0
    if position_buy == 'buy':
        order_state = 1
    elif position_sell == 'sell':
        order_state = 2
    state = np.append(state/100000, [order_state])
    return state

def act(state):
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = model(state)
    q_values = q_values.cpu().numpy()[0]
    # Choose the action with the highest Q-value
    action = np.argmax(q_values)
    return action

def execute_action(action):
    global position_buy, position_sell, entry_price_buy, entry_price_sell
    global wait_buy, wait_sell, profit_buy, profit_sell, previous_profit_buy, previous_profit_sell
    global drawdown, drawup

    balance = get_account_balance()
    update_positions()

    # Get current prices
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print("Failed to get tick for", symbol)
        return
    current_price_buy = tick.ask  # For buy orders
    current_price_sell = tick.bid  # For sell orders
    # Update profits
    if position_buy == 'buy':
        profit_buy = current_price_sell - entry_price_buy  # Use bid price for potential closing
    else:
        profit_buy = 0.0
    if position_sell == 'sell':
        profit_sell = entry_price_sell - current_price_buy  # Use ask price for potential closing
    else:
        profit_sell = 0.0
    # Update drawdown and drawup
    profit_tmp = profit_buy + profit_sell
    if drawdown > profit_tmp:
        drawdown = profit_tmp
    if drawup < profit_tmp:
        drawup = profit_tmp
    # Execute action
    if action == 1:  # Buy
        if position_buy is None:
            # Open buy position
            open_buy_order()
            position_buy = 'buy'
            entry_price_buy = current_price_buy
            wait_buy = 0
            print("Open buy position !!!")
        else:
            print("Cannot open buy position, already in position")
    elif action == 2:  # Sell
        if position_sell is None:
            # Open sell position
            open_sell_order()
            position_sell = 'sell'
            entry_price_sell = current_price_sell
            wait_sell = 0
            print("Open sell position !!!")
        else:
            print("Cannot open sell position, already in position")
    elif action == 0 or action == 3:  # Wait
        wait_buy += 1
        wait_sell += 1
        print("Wait...")
    # Update previous profits
    previous_profit_buy = profit_buy
    previous_profit_sell = profit_sell

def open_buy_order():
    # Récupérer le solde du compte
    balance = get_account_balance()
    if balance is None:
        print("Cannot retrieve account balance")
        return
    # Prepare the request
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(symbol, "not found")
        return
    if not symbol_info.visible:
        if not mt5.symbol_select(symbol, True):
            print("Failed to select", symbol)
            return
    # Calculate lot size based on balance
    if balance <= 1000:
        lot = 0.01
    elif balance <= 10000:
        lot = 0.1
    elif balance <= 100000:
        lot = 1.0
    elif balance <= 1000000:
        lot = 10.0
    elif balance <= 10000000:
        lot = 100.0
    price = mt5.symbol_info_tick(symbol).ask
    point = symbol_info.point
    sl_price = price - 3000 * point  # SL en dessous du prix d'entrée
    tp_price = price + 30000 * point
    deviation = 20
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": mt5.ORDER_TYPE_BUY,
        "price": price,
        "sl": sl_price,
        "tp": tp_price,
        "deviation": deviation,
        "magic": 234000,  # Arbitrary number
        "comment": "Python script open",
        "type_time": mt5.ORDER_TIME_GTC,  # Good till canceled
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    # Send the order
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print("Order send failed, retcode =", result.retcode)
    else:
        print("Buy order opened at price", price)

def open_sell_order():
    # Récupérer le solde du compte
    balance = get_account_balance()
    if balance is None:
        print("Cannot retrieve account balance")
        return
    # Prepare the request
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(symbol, "not found")
        return
    if not symbol_info.visible:
        if not mt5.symbol_select(symbol, True):
            print("Failed to select", symbol)
            return
        
    if balance <= 1000:
        lot = 0.01
    elif balance <= 10000:
        lot = 0.1
    elif balance <= 100000:
        lot = 1.0
    elif balance <= 1000000:
        lot = 10.0
    elif balance <= 10000000:
        lot = 100.0

    price = mt5.symbol_info_tick(symbol).bid
    point = symbol_info.point
    sl_price = price + 3000 * point  # SL au-dessus du prix d'entrée
    tp_price = price - 30000 * point  
    deviation = 20
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": sl_price,
        "tp": tp_price,
        "deviation": deviation,
        "magic": 234000,  # Arbitrary number
        "comment": "Python script open sell",
        "type_time": mt5.ORDER_TIME_GTC,  # Good till canceled
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    # Send the order
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print("Order send failed, retcode =", result.retcode)
    else:
        print("Sell order opened at price", price)

def close_buy_order():
    positions = mt5.positions_get(symbol=symbol)
    if positions is None or len(positions) == 0:
        print("No positions to close")
        return
    for pos in positions:
        if pos.type == mt5.POSITION_TYPE_BUY:
            lot = pos.volume
            price = mt5.symbol_info_tick(symbol).bid
            deviation = 20
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot,
                "type": mt5.ORDER_TYPE_SELL,
                "position": pos.ticket,
                "price": price,
                "deviation": deviation,
                "magic": 234000,
                "comment": "Python script close buy",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                print("Failed to close buy position, retcode =", result.retcode)
            else:
                print("Buy position closed at price", price)
            break

def close_sell_order():
    positions = mt5.positions_get(symbol=symbol)
    if positions is None or len(positions) == 0:
        print("No positions to close")
        return
    for pos in positions:
        if pos.type == mt5.POSITION_TYPE_SELL:
            lot = pos.volume
            price = mt5.symbol_info_tick(symbol).ask
            deviation = 20
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot,
                "type": mt5.ORDER_TYPE_BUY,
                "position": pos.ticket,
                "price": price,
                "deviation": deviation,
                "magic": 234000,
                "comment": "Python script close sell",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                print("Failed to close sell position, retcode =", result.retcode)
            else:
                print("Sell position closed at price", price)
            break

def main():
    try:
        account_info = mt5.account_info()
        if account_info is None:
            print("Failed to get account info")
            return
        if account_info.trade_mode != mt5.ACCOUNT_TRADE_MODE_DEMO:
            print("Warning: Trading on a live account")
        print("Account Info:", account_info)

        # Afficher le solde initial du compte
        balance = get_account_balance()
        if balance is not None:
            print(f"Initial Account Balance: {balance}")
        
        update_positions()

        while True:
            # Récupérer et afficher le solde du compte à chaque itération
            balance = get_account_balance()
            if balance is not None:
                print(f"Current Account Balance: {balance}")
            update_positions()

            # Wait for a new bar
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 1, 1)
            if rates is None or len(rates) == 0:
                print("Failed to get rates")
                time.sleep(10)
                continue
            last_bar_time = rates[0]['time']
            while True:
                update_positions()
                time.sleep(1)  # Wait for 1 second
                rates = mt5.copy_rates_from_pos(symbol, timeframe, 1, 1)
                if rates is None or len(rates) == 0:
                    continue
                new_last_bar_time = rates[0]['time']
                if new_last_bar_time > last_bar_time:
                    break
            # Now a new bar has formed
            state = get_state()
            if state is None:
                continue
            # Use the model to predict an action
            action = act(state)
            # Execute the action
            execute_action(action)
    except KeyboardInterrupt:
        print("Script interrupted by user")
    finally:
        # Close any open positions
        if position_buy == 'buy':
            close_buy_order()
        if position_sell == 'sell':
            close_sell_order()
        # Shutdown MT5 connection
        mt5.shutdown()
        print("MT5 shutdown")

if __name__ == "__main__":
    main()
