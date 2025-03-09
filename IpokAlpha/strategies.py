from ta.utils import dropna
from ta.volatility import BollingerBands
from ta.trend import ADXIndicator
from ta.volatility import AverageTrueRange
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator
from ta.volume import VolumeWeightedAveragePrice

from IpokAlpha.main_functions import IpokSimpleBacktester, IpokOptimiser
import numpy as np
import pandas as pd


class MACrossover(IpokSimpleBacktester):
    def __init__(self, symbol, start, end, interval):
        super().__init__(symbol, start, end, interval)

    def strategy(self, STMA_window, LTMA_window):
        bt_data = self.data.copy()
        indicator_1 = SMAIndicator(close=bt_data["Close"], window=STMA_window, fillna=False)
        STMA = indicator_1.sma_indicator()
        indicator_2 = SMAIndicator(close=bt_data["Close"], window=LTMA_window, fillna=False)
        LTMA = indicator_2.sma_indicator()

        bt_data["STMA"] = STMA
        bt_data["LTMA"] = LTMA
        bt_data = bt_data.dropna()

        bt_data["Position"] = np.where(bt_data['STMA'] > bt_data['LTMA'], 1.0, -1.0)
        bt_data["Signal"] = bt_data['Position'].diff()
        bt_data = bt_data.dropna()

        self.bt_data = bt_data

        return bt_data


class RSIHighLow(IpokSimpleBacktester):
    def __init__(self, symbol, start, end, interval):
        super().__init__(symbol, start, end, interval)


    def strategy(self, upper_rsi=70, lower_rsi=30, rsi_window=14):
        bt_data = self.data.copy()
        indicator = RSIIndicator(close=bt_data["Close"], window=rsi_window, fillna=False)
        RSI = indicator.rsi()
        bt_data["RSI"] = RSI
        
        # Initialize Signal and Position columns
        bt_data['Signal'] = 0
        bt_data['Position'] = 0
        
        # Generate signals with holding constraints
        signals = []
        signal = 0
        holding = 0  # Track position state
        for i in range(len(bt_data)):
            if bt_data['RSI'].iloc[i] > upper_rsi and holding != -1:
                if holding == 0:
                    signal = -1
                else:
                    signal = -2
                signals.append(signal)  # Sell only if holding
                holding = -1
            elif bt_data['RSI'].iloc[i] < lower_rsi and holding != 1:
                if holding == 0:
                    signal = 1
                else:
                    signal = 2
                signals.append(signal)  # Buy only if not holding
                holding = 1
            else:
                signals.append(0)  # No trade
        
        bt_data['Signal'] = signals
        
        # Compute position based on signals
        bt_data['Position'] = np.cumsum(bt_data['Signal']).clip(-1, 1)

        self.bt_data = bt_data
        return bt_data


class ADXDI(IpokSimpleBacktester):
    def __init__(self, symbol, start, end, interval):
        super().__init__(symbol, start, end, interval)

    def strategy(self, adx_threshold=25, di_threshold=20):
        bt_data = self.data.copy()

        adx_indicator = ADXIndicator(bt_data['High'], bt_data['Low'], bt_data['Close'], window=14)

        bt_data['adx'] = adx_indicator.adx()
        bt_data['plus_di'] = adx_indicator.adx_pos()
        bt_data['minus_di'] = adx_indicator.adx_neg()

        signals = []
        holding = 0  # Track position state
        for i in range(len(bt_data)):
            if bt_data['adx'].iloc[i] > adx_threshold:
                if bt_data['plus_di'].iloc[i] > bt_data['minus_di'].iloc[i] and bt_data['plus_di'].iloc[i] > di_threshold and holding != 1:
                    if holding == 0:
                        signal = 1
                    else:
                        signal = 2 
                    holding = 1
                    signals.append(signal) # buy
                elif bt_data['minus_di'].iloc[i] > bt_data['plus_di'].iloc[i] and bt_data['minus_di'].iloc[i] > di_threshold and holding != -1:
                    if holding == 0:
                        signal = -1
                    else:
                        signal = -2
                    holding = -1
                    signals.append(signal) # Sell
                else:
                    signals.append(0)
            else:
                signals.append(0)

        bt_data["Signal"] = signals
        bt_data['Position'] = np.cumsum(bt_data['Signal']).clip(-1, 1)

        self.bt_data = bt_data
        return bt_data