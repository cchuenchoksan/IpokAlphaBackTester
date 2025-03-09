# import all libraries

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from tqdm import tqdm

import warnings

# Making a class for the strategy
# init method to construct and provide all the variables that will be required by the strategy

class IpokSimpleBacktester():

    def __init__(self, symbol, start, end, interval, data=None):

        self.symbol = symbol
        self.start = start
        self.end = end
        self.interval = interval
        #self.results = None
        if data is None:
            self._download_data()
        else:
            self.data = data


    def _download_data(self):
        """Private method to download data from yfinance"""
        stock_data = yf.Ticker(self.symbol)
        data = stock_data.history(start=self.start, end=self.end, interval=self.interval)
        self.data = data
        return data


    # --------------OVERWRITE THIS FUNCTION FOR EACH STRATEGY-------------

    def strategy(self, *args, **kwargs):
        """
        This method builds a `Position` column and `Signal` column of a strategy. 
        To be overwrite by your own strategy with the following definition:
        

        1. Position: A column where 1 means long position and -1 means short position and 0 means you are not holding that stock.
        2. Signal: A column where a positive number means a buy, a negative number means a sell, and a 0 means no trade.
        """
        warnings.warn("YOU ARE RUNNING THE STRATGY FUNCTION FROM THE IpokSimpleBackTester WHICH SHOULD HAVE BEEN OVERIDDEN BY ANOTHER CLASS")
        bt_data = self.data.copy()
        bt_data["Position"] = 0
        bt_data["Signal"] = bt_data['Position'].diff()
        self.bt_data = bt_data
        
        return self.bt_data
    
    # -------------------------------------------------------------------


    def get_returns(self):
        """Get the return of stocks and return of the strategy"""
        self.bt_data['Stock_Returns'] = np.log(self.bt_data["Close"] / self.bt_data["Close"].shift(1))
        self.bt_data["Strategy_Returns"] = self.bt_data["Stock_Returns"] * self.bt_data["Position"].shift(1)
        return self.bt_data


    def visualise_strategy(self, title="Stock price with buy and sell locations and volume in portfolio"):
        """Plot the stock price over time with the position of each trade. Run your strategy function first."""

        if "Signal" not in self.bt_data.columns:
            warnings.warn("The Signal column was not found while attempting to plot. Will re-run self.strategy.")
            self.strategy()
            if "Signal" not in self.bt_data.columns:
                raise KeyError("Cannot find the Signal column in the data")

        fig, ax1 = plt.subplots()
        fig.set_size_inches(14, 7, forward=True)

        ax1.plot(self.bt_data['Close'].index, self.bt_data['Close'], label='Stock Close Price', color='blue', alpha=0.5)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price')
        
        buy_trades = self.bt_data[self.bt_data['Signal'] > 0]
        sell_trades = self.bt_data[self.bt_data['Signal'] < 0]

        ax1.scatter(buy_trades.index, self.bt_data['Close'][buy_trades.index], marker='^', color='g', label='Buy', s=50)
        ax1.scatter(sell_trades.index, self.bt_data['Close'][sell_trades.index], marker='v', color='r', label='Sell', s=50)

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels1 + labels2, handles1 + handles2))

        plt.title(title)
        plt.legend(by_label.values(), by_label.keys())
        plt.grid()
        plt.show()


    def print_performance(self):
        """
        Print the performance metrics:
        1. Annual regular returns
        2. Annual regular standard deviation
        3. The Sharpe Ratio
        """

        if "Stock_Returns" not in self.bt_data.columns or "Strategy_Returns" not in self.bt_data.columns:
            self.get_returns()

        print("----------------------------------------")
        print("The Performance metrics")
        print("----------------------------------------")

        bt_data = self.bt_data

        daily_ret = bt_data[["Stock_Returns", "Strategy_Returns"]].mean()
        annual_ret =  daily_ret * 252
        annual_regular_ret = np.exp(annual_ret)-1

        print("Annual regular returns:")
        print( annual_regular_ret)
        print("----------------------------------------")

        daily_std = bt_data[["Stock_Returns", "Strategy_Returns"]].std()
        annual_std =  daily_std * (252 **0.5)
        daily_regular_std = (np.exp(bt_data[["Stock_Returns", "Strategy_Returns"]])-1).std()
        annual_regular_std =  daily_regular_std * (252 **0.5)


        print('Annual regular standard deviation:')
        print(annual_regular_std)
        print("----------------------------------------")

        sr = annual_regular_ret/ annual_regular_std

        print("The Sharpe Ratio:")
        print(sr)
        print("----------------------------------------")


    def _performance(self):
        """
        Get the performance metrics:
        1. Annual regular returns
        2. Annual regular standard deviation
        3. The Sharpe Ratio
        """

        if "Stock_Returns" not in self.bt_data.columns or "Strategy_Returns" not in self.bt_data.columns:
            self.get_returns()

        bt_data = self.bt_data
        daily_ret = bt_data[["Strategy_Returns"]].mean()
        annual_ret =  daily_ret * 252
        annual_regular_ret = np.exp(annual_ret)-1
        daily_std = bt_data[["Strategy_Returns"]].std()
        annual_std =  daily_std * (252 **0.5)
        daily_regular_std = (np.exp(bt_data[["Strategy_Returns"]])-1).std()
        annual_regular_std =  daily_regular_std * (252 **0.5)
        sr = annual_regular_ret/ annual_regular_std
        return float(annual_regular_ret.iloc[0]), float(annual_regular_std.iloc[0]), float(sr.iloc[0])


    def returns_plot(self, title="Returns plot"):
        """Plot return of the stock vs the strategy"""

        if "Stock_Returns" not in self.bt_data.columns or "Strategy_Returns" not in self.bt_data.columns:
            self.get_returns()
        
        bt_data = self.bt_data.copy()
        bt_data[["Stock_Returns", "Strategy_Returns"]].cumsum().apply(np.exp).plot(title = title, figsize=(15,6))
        plt.show()


    def drawdown(self, title = "Drawdown plot"):
        if "Stock_Returns" not in self.bt_data.columns or "Strategy_Returns" not in self.bt_data.columns:
            self.get_returns()
       
        bt_data = self.bt_data.copy()
        bt_data["Gross_Cum_Returns"] = bt_data["Strategy_Returns"].cumsum().apply(np.exp)  # Same as what we did earlier to visulaise over time
        bt_data["Cum_Max"] = bt_data["Gross_Cum_Returns"].cummax()

        bt_data[["Gross_Cum_Returns", "Cum_Max"]].dropna().plot(title = title, figsize =(15,6))

        drawdown = bt_data["Cum_Max"] - bt_data["Gross_Cum_Returns"]

        print ("The maximum drawdown of the strategy is" , drawdown.max() )
        print("----------------------------------------")

        zero_periods = drawdown[drawdown == 0]
        delta_values = zero_periods.index[1:] - zero_periods.index[:-1]

        print( "The maximum period of drawdown is", delta_values.max() )
        print("----------------------------------------")


class IpokOptimiser():
    def __init__(self, strategy: IpokSimpleBacktester, param_grid=None, params=None):
        self.strategy = strategy

        if params is None and param_grid is None:
            raise ValueError("Need a param_grid or params")

        if params is None:
            self.params = self.get_grid(param_grid)
        else:
            self.params = params

    def fit(self):
        print(f"Testing {len(self.params)} parameters...\n")
        best_param = None
        best_performance = {"returns": 0, "std": 0, "sr": 0}
        for param in self.params:
            self.strategy.strategy(*param)
            ret, std, sr = self.strategy._performance()
            if ret > best_performance["returns"]:
                best_param = param
                best_performance = {"returns": ret, "std": std, "sr": sr}
        
        self.best_param = best_param
        self.best_performance = best_performance
        return best_param

    def get_grid(self):
        pass