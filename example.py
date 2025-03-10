from IpokAlpha.main_functions import IpokOptimiser
from IpokAlpha.strategies import RSIHighLow, MACrossover, ADXDI
import numpy as np


# ipok_rsi = RSIHighLow("AAPL", "2022-01-01", "2023-12-31", "1D")
# ipok_rsi.strategy()
# ipok_rsi.visualise_strategy()
# ipok_rsi.print_performance()
# ipok_rsi.returns_plot()
# ipok_rsi.drawdown()

# ipok_ma = MACrossover("AAPL", "2022-01-01", "2023-12-31", "1D")
# ipok_ma.strategy(5, 50)
# ipok_ma.visualise_strategy()
# ipok_ma.print_performance()
# ipok_ma.returns_plot()
# ipok_ma.drawdown()

# ipok_adx = ADXDI("AAPL", "2022-01-01", "2023-12-31", "1D")
# ipok_adx.strategy(25, 20)
# ipok_adx.visualise_strategy()
# ipok_adx.print_performance()
# ipok_adx.returns_plot()
# ipok_adx.drawdown()

# ---------------------------------------------------------------------------------------------------------------

# Optimiser
params = []
for i in range(10, 100):
    for j in range(i+5, 110):
        params.append([i, j])

ipok_ma = MACrossover("TSLA", "2022-01-01", "2023-12-31", "1D")
optimiser = IpokOptimiser(ipok_ma, params=params)
best_param = optimiser.fit()
print(optimiser.best_param)
print(optimiser.best_performance)

ipok_ma.strategy(best_param[0], best_param[1])
ipok_ma.visualise_strategy()
ipok_ma.print_performance()
ipok_ma.returns_plot()
ipok_ma.drawdown()

# ---------------------------------------------------------------------------------------------------------------

# params = []
# for i in np.arange(15, 40, 0.5):
#     for j in np.arange(60, 85, 0.5):
#         params.append([i, j])

# ipok_ma = RSIHighLow("AAPL", "2022-01-01", "2023-12-31", "1D")
# optimiser = IpokOptimiser(ipok_ma, params=params)
# best_param = optimiser.fit()
# print(optimiser.best_param)
# print(optimiser.best_performance)

# ipok_ma.strategy(best_param[0], best_param[1])
# ipok_ma.visualise_strategy()
# ipok_ma.print_performance()
# ipok_ma.returns_plot()
# ipok_ma.drawdown()