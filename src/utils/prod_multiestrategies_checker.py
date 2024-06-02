from datetime import datetime, timedelta
import pandas as pd
import  matplotlib.pyplot as plt
import numpy as np
import pytz

import warnings
warnings.filterwarnings('ignore')

import requests

from utils.recommendations import BrainyRecommendationsRepository
from utils.prices import PricesRepository, CandlestickRepository
from utils.recommendations_runner import RecommendationsRunner


reco_repository = BrainyRecommendationsRepository(target_url='http://10.10.26.85/recommendations/')
prices_repository = CandlestickRepository.default_repository()



confs_modification = [
    (datetime(2021, 1, 17).replace(tzinfo=pytz.utc), "Apagamos el perfil BTC", 'r'),
    (datetime(2021, 1, 19, 17).replace(tzinfo=pytz.utc), "NowBetterThanBefore Strategies - Parametros 001", 'g'),
    (datetime(2021, 1, 20, 20).replace(tzinfo=pytz.utc), "NowBetterThanBefore Strategies - Parametros 002", 'g'),
    (datetime(2021, 1, 22, 15).replace(tzinfo=pytz.utc), "NowBetterThanBefore Strategies - Francu, Mary y Chun", 'g'),
    (datetime(2021, 1, 25, 16).replace(tzinfo=pytz.utc), "By I+D", 'g'), 
    (datetime(2021, 1, 27, 15).replace(tzinfo=pytz.utc), "By I+D", 'g'),     
    (datetime(2021, 1, 29, 15).replace(tzinfo=pytz.utc), "By I+D", 'g'),     
    (datetime(2021, 1, 30, 11).replace(tzinfo=pytz.utc), "By I+D", 'g'),     
    (datetime(2021, 2, 1, 13).replace(tzinfo=pytz.utc), "By I+D", 'g'),     
    (datetime(2021, 2, 2, 18).replace(tzinfo=pytz.utc), "SL + TP - Semana 0", 'b'), 
    (datetime(2021, 2, 8, 17).replace(tzinfo=pytz.utc), "Semana 1", 'b'),         
    (datetime(2021, 2, 15, 17).replace(tzinfo=pytz.utc), "Semana 2", 'b'),        
    (datetime(2021, 2, 22, 17).replace(tzinfo=pytz.utc), "Semana 3", 'b'),        
    (datetime(2021, 3, 1, 17).replace(tzinfo=pytz.utc), "Semana 4", 'b'),        
    (datetime(2021, 3, 8, 12).replace(tzinfo=pytz.utc), "Semana 5", 'b'),        
    (datetime(2021, 3, 15, 12).replace(tzinfo=pytz.utc), "Semana 6", 'b'),          
    (datetime(2021, 3, 22, 12).replace(tzinfo=pytz.utc), "Semana 7", 'b'),          
    (datetime(2021, 3, 29, 12).replace(tzinfo=pytz.utc), "Semana 8", 'b'),          
    (datetime(2021, 4, 5, 12).replace(tzinfo=pytz.utc), "Semana 9", 'b'),          
    (datetime(2021, 4, 12, 12).replace(tzinfo=pytz.utc), "Semana 10", 'b'),          
#    (datetime(2021, 3, 15, 12).replace(tzinfo=pytz.utc), "Semana 5", 'b'),          
]

def calc_strategy_returns(trades_df):
    trades_df = trades_df.copy()
    trades_df["strategy_return"] = np.where(trades_df.holding.shift(1), trades_df["asset_return"],0)
    commission_condition = trades_df.holding.shift(1)!=trades_df.holding.shift(2)
    trades_df["strategy_return"] = np.where(commission_condition, trades_df["strategy_return"]-0.001, trades_df["strategy_return"])
    return trades_df

def count_trades(trades_df):
    trades_df['trade_number'] = np.nan

    if trades_df.strategy_return.iloc[0] != 0:
        counter = 1
        trades_df['trade_number'].iloc[0] = 1
    else:
        counter = 0
        
    for i in range(1, len(trades_df)):
        if trades_df.strategy_return.iloc[i-1] == 0 and trades_df.strategy_return.iloc[i] != 0:
            counter += 1

        if trades_df.strategy_return.iloc[i] != 0:
            trades_df['trade_number'].iloc[i] = counter
    
    return trades_df

def create_trades_df(asset_code, quote_code, full_recos):
    if asset_code == 'USDT' and quote_code == 'BTC':
        candles = prices_repository.get_candlestick(
            pair=f'{quote_code}/{asset_code}',
            exchange='binance', 
            size=60,
            start_time=full_recos.index[0], 
            end_time=full_recos.index[-1],
        )
        _open = 1/candles.open
    else:
        candles = prices_repository.get_candlestick(
            pair=f'{asset_code}/{quote_code}',
            exchange='binance', 
            size=60,
            start_time=full_recos.index[0], 
            end_time=full_recos.index[-1],
        )
        _open = candles.open
    
    trades_df = pd.DataFrame({
        'pair': f'{asset_code}/{quote_code}',
        'open': _open,
        'asset_return': _open.pct_change().fillna(0),
        'holding': full_recos[asset_code] > 0,
    })
    
    trades_df.dropna(inplace=True)
    trades_df = calc_strategy_returns(trades_df)
    trades_df = count_trades(trades_df)
    
    return trades_df

def process_trades(counted_trades):
    data = []
    for trade_number, trade_df in counted_trades.dropna().groupby('trade_number'):
        data.append({
            'trade_number': trade_number,
            'performance': (trade_df['strategy_return'] + 1).cumprod()[-1]-1,
            'trade_duration': trade_df.index[-1] - trade_df.index[0],
            'start_time': trade_df.index[0],
            'end_time': trade_df.index[-1],
        })
    df = pd.DataFrame(data) 
    return df


def plot_strategy_returns(trades_df):
    performance = (trades_df[['asset_return', 'strategy_return']] + 1).cumprod()
    strategy_return = performance.iloc[-1]['strategy_return'] - 1
    asset_return = performance.iloc[-1]['asset_return'] - 1
    print(f"Performance strategy over {trades_df.pair.iloc[0]}: {strategy_return} (asset return {asset_return})")
    
    plt.figure(figsize=(20, 6))
    plt.plot(performance.asset_return, label='Asset Return')
    plt.plot(performance.strategy_return, label="Strategy Return")
    
    for timestamp, label, color in confs_modification:
        plt.axvline(x=timestamp, label=label, c=color, linestyle='--')

    plt.grid(True)        
    plt.legend(loc="best")
    plt.tight_layout()    
    plt.show()

    
    # Secod Graphic
    plt.figure(figsize=(20, 6))
    last_conf = confs_modification[1][0]
    restricted_returns = performance[['asset_return', 'strategy_return']][last_conf:].pct_change()
    restricted_returns.iloc[0] = 0    
    
    plt.plot((restricted_returns.asset_return + 1).cumprod(), label='Asset Return since last change')
    plt.plot((restricted_returns.strategy_return + 1).cumprod(), label="Strategy Return since last change")

    for timestamp, label, color in confs_modification[1:]:
        plt.axvline(x=timestamp, label=label, c=color, linestyle='--')

    plt.grid(True)
    plt.legend(loc="best")
    plt.tight_layout()    
    
    plt.show()

    
def display_full_data(asset_code, quote_code, full_recos):
    trades = create_trades_df(asset_code, quote_code, full_recos)
    plot_strategy_returns(trades)
    display(process_trades(trades))
    return trades


from datetime import timedelta

def get_portfolio_returns(recommendations, currency, prices):
    reco_runner = RecommendationsRunner()
    _, portfolio_returns = reco_runner.run(
        currency, 
        recommendations[~recommendations.index.duplicated(keep='first')], 
        prices)

    return portfolio_returns

def plot_portfolio_returns(portfolio_returns):
    plt.figure(figsize=(22, 8))
    plt.plot((portfolio_returns.fillna(0) + 1).cumprod(), label='Portfolio return')
    
    for timestamp, label, color in confs_modification:
        plt.axvline(x=timestamp, label=label, c=color, linestyle='--')

    plt.legend(loc="upper left")
    plt.grid(True)
    plt.tight_layout()    
    plt.show()

    plt.figure(figsize=(22, 8))
    last_conf = confs_modification[1][0]
    plt.plot((portfolio_returns[last_conf:].fillna(0) + 1).cumprod(), label='Portfolio return')
    
    for timestamp, label, color in confs_modification[1:]:
        plt.axvline(x=timestamp, label=label, c=color, linestyle='--')

    plt.grid(True)
    plt.legend(loc="best")
    plt.tight_layout()    
    plt.show()

    
N = len(confs_modification) - 8
semanas = confs_modification[-N:]
def print_semanal_performance(returns):
    pares = [(semanas[i][0],semanas[i+1][0]) for i in range(len(semanas) - 1)]
    pares.append((semanas[-1][0], datetime.utcnow().replace(tzinfo=pytz.utc) ))
    
    for start, end in pares:
        r = round((returns.loc[start:end] + 1).cumprod().iloc[-1] - 1, 3)
        print(f"semana del {start}: {r}")
        
