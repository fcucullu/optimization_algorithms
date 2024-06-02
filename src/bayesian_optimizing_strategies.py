# By FranCu

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from utils.candlestick import CandlestickRepository
from utils.xmetrics import Xmetrics
import utils.simulations as sim
from utils.strategies import TwoStandardMovingAverage, ThreeStandardMovingAverage, ThreeStandardMovingAverageAlternative, ChandelierExitStrategy, VWAPvsSMA, BBvolumeStrategy
from utils.optimizators import bayes_optimize

###############################################################################
'''                                    INPUTS                               '''
###############################################################################

Xmetrics = Xmetrics()
Repository = CandlestickRepository.default_repository()

###PERIOD TOP STUDY !###
start_date = datetime(2020,12,15)
end_date = datetime(2022,1,1)
ticker = 'BTC/USDT'
exchange = 'binance'
###PERIOD TOP STUDY !###


minutes = 60*1
candles_1h = Repository.get_candlestick(ticker,exchange,minutes,start_date,end_date)
candles_1h = sim.FillNa().fill_ohlc(candles_1h)    

minutes = 60*4
candles_4h = Repository.get_candlestick(ticker,exchange,minutes,start_date,end_date)
candles_4h = sim.FillNa().fill_ohlc(candles_4h)

minutes = 60*24
candles_24h = Repository.get_candlestick(ticker,exchange,minutes,start_date,end_date)
candles_24h = sim.FillNa().fill_ohlc(candles_24h)        

###############################################################################
'''                    two_moving_average_strategy                          '''
###############################################################################

strategy = TwoStandardMovingAverage(candles_4h)  
bounds = strategy.get_standard_bounds()



methods = {'upper confidece bound': 'ucb',
           'expected improvement': 'ei',
           'probability of improvement': 'poi'}

for k,v in methods.items():
    results = bayes_optimize(strategy, 
                             bounds, 
                             v)
    df = df.append(results)
    df['best'] = 0
    df['best'] = np.where(df.performance == df.performance.max(), 1, 0)

results['week'] = [123]

###############################################################################
'''                    three_moving_average_strategy                        '''
###############################################################################

strategy = ThreeStandardMovingAverage(candles_4h)
func = strategy.get_performance    

results = bayes_optimize(func, bounds, True)
print(pd.DataFrame.from_dict(results.max))

###############################################################################
'''                    three_moving_average_strategy v2                     '''
###############################################################################

strategy = ThreeStandardMovingAverageAlternative(candles_4h)
func = strategy.get_performance    

results = bayes_optimize(func, bounds,True)
print(pd.DataFrame.from_dict(results.max))

###############################################################################
'''                    Chandelier Exits Strategy                           '''
###############################################################################

strategy = ChandelierExitStrategy(candles_4h)
func = strategy.get_performance    

results = bayes_optimize(func, bounds, True)
print(pd.DataFrame.from_dict(results.max))

###############################################################################
'''                         VWAP vs SMA                                    '''
###############################################################################


strategy = VWAPvsSMA(candles_4h)
func = strategy.get_performance    

results = bayes_optimize(func, bounds, True)
print(pd.DataFrame.from_dict(results.max))

###############################################################################
'''                         BBvolumeStrategy                                '''
###############################################################################


strategy = BBvolumeStrategy(candles_4h)
func = strategy.get_performance    

results = bayes_optimize(func, bounds, True)
print(pd.DataFrame.from_dict(results.max))


# ###############################################################################
# '''                 COMPARACION  DE  METRICAS                               '''
# ###############################################################################
# plt.figure(figsize=(12,10))
# plt.plot((candles.open.pct_change()+1).cumprod(), label="Holding")
# plt.plot(nbtb.performance, label="NBTB")
# plt.plot(d_nbtb.performance, label="D_NBTB")
# plt.plot(d_nbtb_ma100.performance, label="D_NBTB_w/MA100")
# plt.plot(tf.performance, label="Trend Follower")
# plt.legend()
# plt.show()

# Xmetrics.strategy_metrics(nbtb, periods, 'signal')
# Xmetrics.strategy_metrics(d_nbtb, periods, 'signal')
# Xmetrics.strategy_metrics(d_nbtb_ma100, periods, 'signal')
# Xmetrics.strategy_metrics(tf, periods, 'signal')

