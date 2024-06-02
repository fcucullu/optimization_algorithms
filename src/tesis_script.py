from utils.candlestick import CandlestickRepository
from datetime import datetime
from utils.xmetrics import Xmetrics
from utils.graphs import Graphs
import utils.simulations as sim
from utils.experiment import Experiment
from utils.strategies import TwoStandardMovingAverage, ThreeStandardMovingAverage, ThreeStandardMovingAverageAlternative, ChandelierExitStrategy, VWAPvsSMA, BBvolumeStrategy
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

Experiment = Experiment()
Graphs = Graphs()
Xmetrics = Xmetrics()
Repository = CandlestickRepository.default_repository()

start_date = datetime(2020,12,1)
end_date = datetime(2022,1,1)
ticker = 'BTC/USDT'
exchange = 'binance'

minutes = 60*4
candles_4h = Repository.get_candlestick(ticker,exchange,minutes,start_date,end_date)
candles_4h = sim.FillNa().fill_ohlc(candles_4h)

n_simulations = 50
periods_in_weeks = 2

strategies = [TwoStandardMovingAverage, 
              ThreeStandardMovingAverage, 
              ThreeStandardMovingAverageAlternative, 
              ChandelierExitStrategy, 
              VWAPvsSMA, 
              BBvolumeStrategy]

strategies = [strategy(candles_4h) for strategy in strategies]

results = Experiment.run_multiple_optimizations_for_multiple_strategies(strategies, n_simulations, periods_in_weeks)







###########################################################################
'''                     Analisis de resultados                 ''' 

import ast
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

dict_results = {}
acq_func =  {'Limite de confianza superior': 'ucb',
             'Mejora esperada': 'ei',
             'Probabilidad de mejora': 'poi'}

for selected_strategy in strategies:
    strategy_name = selected_strategy.__name__
    data_results = pd.DataFrame()
    for real_name, function in acq_func.items():        
        candles_result = candles_4h.copy()
        candles_result = candles_result[datetime(2021,1,1):]
        candles_result['week'] = candles_result.index.isocalendar().week
        candles_result['signal'] = 0
        
        list_of_candles = Experiment.divide_candles_in_weeks(candles_4h, 2)
        
        
        for candles_slice in list_of_candles[1:]:
            week = candles_slice.week[-1]
    
            param_results = pd.read_csv(f'C:\\Users\\Francisco\\Desktop\\Facultad\\Maestria en Finanzas Quantitativas\\TESIS\\REDACCION\\2do intento\\src\\results_{strategy_name}_20sim_2week.csv')
            param_results = param_results.loc[param_results.acq_func == function]
            param_results = param_results.loc[param_results.week == week]
            param_results = ast.literal_eval(param_results.loc[param_results.week == week, 'params'].values[0])

            strategy = selected_strategy(candles_slice.copy())
            results_slice = strategy.process_candles(**param_results)

            candles_result.loc[candles_result.week == week, 'signal'] = results_slice.loc[results_slice.week == week, 'signal']
            
        candles_result = Xmetrics.calculate_returns(candles_result)
        candles_result['performance'] = (candles_result['returns']+1).cumprod()
        data_results[f'{real_name}'] = candles_result['performance']
        
    dict_results.update({strategy_name: data_results.copy()})

for k,v in dict_results.items():
    fig=plt.figure(figsize=(12, 7))
    fig.suptitle(k, fontsize=22, fontweight="bold")
    plt.plot(dict_results[k], lw=2, alpha=0.8)
    plt.xlabel('Tiempo',fontsize=15)
    plt.ylabel('Performance',fontsize=15)
    plt.legend(v,loc=2, fontsize=15)
    plt.show()    
    
    
for selected_strategy in strategies:
    strategy_name = selected_strategy.__name__
    param_results = pd.read_csv(f'C:\\Users\\Francisco\\Desktop\\Facultad\\Maestria en Finanzas Quantitativas\\TESIS\\REDACCION\\2do intento\\src\\results_{strategy_name}_20sim_2week.csv')
    
    fig=plt.figure(figsize=(12, 7))
    fig.suptitle(strategy_name, fontsize=22, fontweight="bold")
    for real_name, function in acq_func.items():
        delays = param_results.loc[param_results.acq_func == function, 'delay']
        sns.distplot(delays, hist=False, kde=True, label=real_name, kde_kws={'linewidth': 4})
    plt.xlabel('Segundos de demora en el cálculo',fontsize=15)
    plt.ylabel('Observaciones',fontsize=15)
    plt.legend(real_name, loc=1, fontsize=15)
    plt.show()    
    
