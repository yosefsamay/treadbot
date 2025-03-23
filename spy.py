import pandas as pd
import numpy as np
from ib_insync import IB, Stock, Option, MarketOrder, LimitOrder, util, Contract
from sklearn.ensemble import GradientBoostingRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import argparse
import warnings
warnings.filterwarnings("ignore")


STRATEGY_SCORE_WEIGHTS = {
    'vertical_bull': 0.8,
    'vertical_bear': 0.8,
    'straddle': 0.6,
    'strangle': 0.5,
    'iron_condor': 0.4,
    'butterfly': 0.6,
    'calendar': 0.5,
    'synthetic_long': 0.7,
    'synthetic_short': 0.7,
    'covered_call': 0.65,
    'protective_put': 0.65
}


def select_best_strategy(forecast_change):
    candidates = []
    for strategy, weight in STRATEGY_SCORE_WEIGHTS.items():
        if forecast_change > 0:
            if 'bear' not in strategy and 'short' not in strategy:
                score = forecast_change * weight
            elif 'synthetic_short' in strategy:
                score = 0  # Not relevant
        elif forecast_change < 0:
            if 'bull' not in strategy and 'long' not in strategy:
                score = abs(forecast_change) * weight
            elif 'synthetic_long' in strategy:
                score = 0  # Not relevant
        elif strategy in ['straddle', 'strangle', 'iron_condor', 'butterfly', 'calendar']:
            score = (abs(forecast_change) * weight) / 2
        else:
            score = 0
        candidates.append((strategy, score))
    best = max(candidates, key=lambda x: x[1])
    return best[0]


def download_prices_ib(ib, symbols, duration='3 Y', barSize='1 day'):
    historical_data = {}
    for symbol in symbols:
        try:
            contract = Stock(symbol, 'SMART', 'USD')
            ib.qualifyContracts(contract)
            bars = ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr=duration,
                barSizeSetting=barSize,
                whatToShow='ADJUSTED_LAST',
                useRTH=True,
                formatDate=1
            )
            df = util.df(bars)
            if not df.empty and df['volume'].mean() > 0:
                df.set_index('date', inplace=True)
                historical_data[symbol] = df['close']
        except Exception as e:
            print(f"⚠️ Failed to download {symbol}: {e}")
    return pd.DataFrame(historical_data).dropna(axis=1)


# Example of how this is integrated into live trading loop:
def trade_with_forecast(ib, symbol, forecast_price, last_price):
    forecast_change = (forecast_price - last_price) / last_price
    selected_strategy = select_best_strategy(forecast_change)
    print(f"🔄 {symbol}: Forecast Change={forecast_change:.4f}, Selected Strategy={selected_strategy}")
    place_option_strategy(ib, symbol, strategy_type=selected_strategy)


def place_option_strategy(ib, symbol, strategy_type):
    print(f"📥 Placing option strategy '{strategy_type}' for {symbol}...")
    # TODO: Implement each strategy's logic here
    if strategy_type == 'vertical_bull':
        print("🟢 Executing Bull Call Spread")
        # Build calls: buy low strike, sell higher strike
        # Implement actual contract selection and submission
    elif strategy_type == 'vertical_bear':
        print("🔴 Executing Bear Put Spread")
    elif strategy_type == 'straddle':
        print("📊 Executing Straddle: Buy Call + Put at same strike")
    elif strategy_type == 'strangle':
        print("📊 Executing Strangle: Buy Call OTM + Put OTM")
    elif strategy_type == 'iron_condor':
        print("📐 Executing Iron Condor")
    elif strategy_type == 'butterfly':
        print("🦋 Executing Butterfly Spread")
    elif strategy_type == 'calendar':
        print("📅 Executing Calendar Spread")
    elif strategy_type == 'synthetic_long':
        print("⚙️ Executing Synthetic Long (Long Call + Short Put)")
    elif strategy_type == 'synthetic_short':
        print("⚙️ Executing Synthetic Short (Short Call + Long Put)")
    elif strategy_type == 'covered_call':
        print("💼 Executing Covered Call (Stock + Short Call)")
    elif strategy_type == 'protective_put':
        print("🛡️ Executing Protective Put (Stock + Long Put)")
    else:
        print("⚠️ Strategy not implemented yet.")


# Add this call inside your main trading loop:
# trade_with_forecast(ib, symbol, forecast_price, last_price)
