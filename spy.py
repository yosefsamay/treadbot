import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from ib_insync import IB, Stock, util

# רשימת 20 מניות עם מתאם נמוך יחסית ממגזרים שונים
low_correlation_symbols = [
    'AAPL', 'JNJ', 'XOM', 'NVDA', 'KO',
    'JPM', 'UNH', 'PG', 'HD', 'PFE',
    'DIS', 'CSCO', 'PEP', 'WMT', 'BAC',
    'MRK', 'T', 'NKE', 'CVX', 'MCD'
]

class EnhancedStockBacktester:
    def __init__(self, data_path=None, use_ib_api=False, symbol='AAPL', port=7496):
        if use_ib_api:
            self.df = self._load_data_from_ib(symbol, port)
        else:
            self.df = self._load_3years_data(data_path)
        self.symbol = symbol
        self.selected_features = None
        self.xgb_model = None
        self.lstm_model = None
        self.scaler = None
        self.sequence_length = 30
        self.forecast_horizon = 5
        self.portfolio = {
            'cash': 100000,
            'positions': {},
            'history': [],
            'total_commissions': 0,
            'total_taxes': 0
        }
        self.commission_per_share = 0.01
        self.min_commission = 2.5
        self.tax_rate = 0.25

    def _load_data_from_ib(self, symbol, port):
        ib = IB()
        ib.connect('127.0.0.1', port, clientId=1)
        contract = Stock(symbol, 'SMART', 'USD')
        bars = ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr='3 Y',
            barSizeSetting='1 day',
            whatToShow='TRADES',
            useRTH=True,
            formatDate=1
        )
        df = util.df(bars)
        ib.disconnect()
        df.rename(columns={'date': 'date', 'close': 'close', 'volume': 'volume'}, inplace=True)
        return df[['date', 'close', 'volume']]

# פונקציה שמריצה את backtest על כל המניות ברשימה

def run_backtests_on_symbols(symbols, port=7496):
    results = []
    for symbol in symbols:
        print(f"\n🚀 מריץ Backtest עבור: {symbol}")
        bt = EnhancedStockBacktester(use_ib_api=True, symbol=symbol, port=port)
        bt.train_model()
        bt.run_backtest()
        results.append((symbol, bt.portfolio['history']))
    return results

# להרצה בפועל (אם רוצים מתוך קובץ או מחוץ ל-Notebook)
# run_backtests_on_symbols(low_correlation_symbols)

# שאר הקוד נשאר כפי שהוא...

    def _prepare_features(self):
        df = self.df.copy()
        df['returns'] = df['close'].pct_change()
        df['MA_10'] = df['close'].rolling(10).mean()
        df['MA_50'] = df['close'].rolling(50).mean()
        df['RSI'] = self._calculate_rsi(df['close'], 14)
        df['MACD'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
        df['Volume_Change'] = df['volume'].pct_change(5)
        df['Volatility'] = df['returns'].rolling(30).std()
        return df.dropna()

    def _calculate_rsi(self, series, window):
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        return 100 - (100 / (1 + (avg_gain / avg_loss)))

    def _select_features(self, X, y):
        model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, n_jobs=-1, tree_method='gpu_hist')
        model.fit(X, y)
        selector = SelectFromModel(model, max_features=7, threshold='median')
        selector.fit(X, y)
        return X.columns[selector.get_support()]

    def _prepare_lstm_data(self, df, features):
        data = df[features].values
        close_prices = df['close'].values
        self.scaler = MinMaxScaler()
        scaled_data = self.scaler.fit_transform(data)

        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data) - self.forecast_horizon):
            X.append(scaled_data[i-self.sequence_length:i])
            future_prices = close_prices[i:i+self.forecast_horizon]
            y.append(future_prices)
        return np.array(X), np.array(y)

    def train_model(self):
        prepared_df = self._prepare_features()
        X = prepared_df.drop(['date', 'close', 'returns'], axis=1, errors='ignore')
        y = prepared_df['close'].shift(-1).dropna()
        X = X.iloc[:-1]
        self.selected_features = self._select_features(X, y)

        self.xgb_model = XGBRegressor(n_estimators=500, learning_rate=0.02, max_depth=5, n_jobs=-1, tree_method='gpu_hist')
        self.xgb_model.fit(X[self.selected_features], y)

        lstm_df = prepared_df[['date', 'close'] + list(self.selected_features)].dropna().reset_index(drop=True)
        X_lstm, y_lstm = self._prepare_lstm_data(lstm_df, self.selected_features)

        self.lstm_model = Sequential([
            LSTM(64, input_shape=(X_lstm.shape[1], X_lstm.shape[2])),
            Dense(self.forecast_horizon)
        ])
        self.lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        self.lstm_model.fit(X_lstm, y_lstm, epochs=10, batch_size=32, verbose=0)

        self.lstm_sequence_df = lstm_df.copy()

    def _predict_lstm(self):
        seq = self.lstm_sequence_df[self.selected_features].values[-self.sequence_length:]
        seq_scaled = self.scaler.transform(seq)
        pred = self.lstm_model.predict(np.expand_dims(seq_scaled, axis=0), verbose=0)[0]
        return pred

    def run_backtest(self):
        prepared_df = self._prepare_features()
        test_data = prepared_df.iloc[-750:].copy().reset_index(drop=True)

        for i in range(self.sequence_length, len(test_data) - self.forecast_horizon):
            row = test_data.iloc[i]
            current_date = row['date']
            current_price = row['close']

            x_row = test_data[self.selected_features].iloc[i].values.reshape(1, -1)
            forecast_xgb = self.xgb_model.predict(x_row)[0]

            self.lstm_sequence_df = test_data.iloc[i-self.sequence_length:i].copy()
            forecast_lstm = self._predict_lstm()

            forecast_3 = forecast_lstm[:3]
            signals = [(p - current_price) / current_price for p in forecast_3]
            up = sum(1 for s in signals if s > 0.01)
            down = sum(1 for s in signals if s < -0.01)

            if up >= 2:
                action = 'buy'
            elif down >= 2:
                action = 'sell'
            else:
                action = 'hold'

            self._execute_trade(current_date, current_price, action)
            self._record_portfolio_value(current_date, current_price)

        self._generate_report()

        def _execute_trade(self, date, price, action):
        position_size = self.portfolio['cash'] * 0.2

        if action == 'buy' and self.portfolio['cash'] >= position_size:
            shares = position_size / price
            commission = self._calculate_commission(shares)
            total_cost = (shares * price) + commission

            if self.portfolio['cash'] >= total_cost:
                self.portfolio['cash'] -= total_cost
                self.portfolio['positions'][date] = {
                    'shares': shares,
                    'entry_price': price,
                    'entry_commission': commission
                }
                self.portfolio['total_commissions'] += commission

        elif action == 'sell' and self.portfolio['positions']:
            for pos_date, pos in list(self.portfolio['positions'].items()):
                exit_commission = self._calculate_commission(pos['shares'])
                gross_profit = (price - pos['entry_price']) * pos['shares']
                tax = self._calculate_tax(gross_profit)
                net_profit = gross_profit - tax - pos['entry_commission'] - exit_commission

                self.portfolio['cash'] += (pos['shares'] * price) - exit_commission - tax
                self.portfolio['total_commissions'] += exit_commission
                self.portfolio['total_taxes'] += tax

                self.portfolio['history'].append({
                    'entry_date': pos_date,
                    'exit_date': date,
                    'shares': pos['shares'],
                    'entry_price': pos['entry_price'],
                    'exit_price': price,
                    'gross_profit': gross_profit,
                    'net_profit': net_profit,
                    'commissions': pos['entry_commission'] + exit_commission,
                    'tax': tax,
                    'holding_days': (date - pos_date).days
                })
                del self.portfolio['positions'][pos_date]

    def _record_portfolio_value(self, date, price):
        positions_value = sum(pos['shares'] * price for pos in self.portfolio['positions'].values())
        self.portfolio['history'].append({
            'date': date,
            'portfolio_value': self.portfolio['cash'] + positions_value
        })

    def _calculate_commission(self, shares):
        return round(max(self.min_commission, shares * self.commission_per_share), 2)

    def _calculate_tax(self, profit):
        return self.tax_rate * max(0, profit)

    def _generate_report(self):
        history_df = pd.DataFrame(self.portfolio['history'])
        trades_df = pd.DataFrame([x for x in self.portfolio['history'] if 'net_profit' in x])

        initial_value = 100000
        final_value = history_df['portfolio_value'].iloc[-1]
        total_net_return = (final_value - initial_value) / initial_value * 100
        annualized_return = (1 + total_net_return/100)**(1/3) - 1

        history_df['peak'] = history_df['portfolio_value'].cummax()
        history_df['drawdown'] = (history_df['portfolio_value'] - history_df['peak']) / history_df['peak']
        max_drawdown = history_df['drawdown'].min() * 100

        win_rate = (trades_df['net_profit'] > 0).mean() * 100 if len(trades_df) > 0 else 0
        avg_profit = trades_df['net_profit'].mean() if len(trades_df) > 0 else 0
        profit_factor = -trades_df[trades_df['net_profit'] > 0]['net_profit'].sum() / \
                        trades_df[trades_df['net_profit'] < 0]['net_profit'].sum() if len(trades_df) > 1 else 0

        print(f"
=== תוצאות Backtest ({self.symbol}) ===")
        print(f"הון התחלתי: ${initial_value:,.2f}")
        print(f"הון סופי: ${final_value:,.2f}")
        print(f"תשואה נטו: {total_net_return:.2f}% (לאחר עמלות ומסים)")
        print(f"תשואה שנתית ממוצעת: {annualized_return*100:.2f}%")
        print(f"מקסימום דרודאון: {max_drawdown:.2f}%")
        print(f"
סה\"כ עמלות: ${self.portfolio['total_commissions']:,.2f}")
        print(f"סה\"כ מסים: ${self.portfolio['total_taxes']:,.2f}")
        print(f"
מדדים סטטיסטיים:")
        print(f"אחוז עסקאות רווחיות: {win_rate:.2f}%")
        print(f"רווח ממוצע לעסקה: ${avg_profit:,.2f}")
        print(f"יחס רווח/הפסד: {profit_factor:.2f}:1")

        if len(trades_df) > 0:
            print("
5 העסקאות האחרונות:")
            print(trades_df[['entry_date', 'exit_date', 'shares', 'net_profit', 'commissions', 'tax']].tail().to_string(index=False))

        if 'date' in history_df.columns and 'portfolio_value' in history_df.columns:
            plt.figure(figsize=(10, 5))
            plt.plot(history_df['date'], history_df['portfolio_value'])
            plt.title(f'שווי תיק ההשקעות לאורך זמן - {self.symbol}')
            plt.xlabel('תאריך')
            plt.ylabel('שווי תיק ($)')
            plt.grid(True)
            plt.show()
