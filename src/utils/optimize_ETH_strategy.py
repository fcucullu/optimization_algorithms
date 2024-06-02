#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 15:49:43 2020

@author: farduh

super optimizador
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from datetime import datetime,timezone,timedelta
from tqdm import tqdm
from candlestick import CandlestickRepository
sys.path.append(r"C:\Users\Francisco\Desktop\Trabajo\XCapit\xcapit_util")
from xcapit_metrics import Xmetrics
sys.path.append(r"C:\Users\Francisco\Desktop\Trabajo\XCapit\estrategia-fran-local\src")
from Estrategia_por_par import BBvolumeStrategy,SignalForceAltStrategy
from bayes_opt import BayesianOptimization
from simulations import MertonJumpDiffusionSimulator,PriceWorker, PriceExplorer, SimulateMLMertonOHLC, SimulateMLGeometricOHLC,Fitter, DistributionGenerator, FillNa
from glob import glob
import json
import quantstats as qs
import itertools
from market_profile import MarketProfile
from market_profile import midmax_idx
import ray
import os

from ta.volume import *
from ta.trend import *
from ta.volatility import *
quotes=["USDT","BTC"]
margin_shorts=[0,1]
dates =[]
date=datetime(2021,9,1)-timedelta(weeks=1)
while date <datetime(2021,11,23):
    dates.append(date)
    date+=timedelta(weeks=1)
    
repo = CandlestickRepository.default_repository()

#generador de simulaciones
def generate_simulation(data,
                        minutes,
                        base,
                        quote,
                        num_sim=1000,
                        sim_length=10000):
        """
        Generate simulations
        """
        if not os.path.exists('samples'):
            os.makedirs('samples')
        print("Getting data")
        
        periods = (60*24*365)/minutes   
        print('Analyzing homoscedasticity')
        explorer, worker = PriceExplorer(), PriceWorker()
        sample = explorer.get_sample_for_simulations_non_normal(data['close'], periods, 0.01)
        sample = worker.rebuild_tabla_by_index(data, sample)
        date_initial,date_final = sample.index[0], sample.index[-1]
        try:
            simulations = pd.read_csv(f"samples/simulation_{date_final.strftime('%Y%m%d')}-{date_initial.strftime('%Y%m%d')}_{base}{quote}_{minutes}t.csv",index_col="Unnamed: 0")
            simulations.index = pd.to_datetime(simulations.index)
            simulations_list = []
            for i in range(int(len(simulations.columns)/5)):
                df=pd.DataFrame(index = simulations.index)
                df["open"] = simulations[f"open_{i}"]
                df["close"] = simulations[f"close_{i}"]
                df["high"] = simulations[f"high_{i}"]
                df["low"] = simulations[f"low_{i}"]
                df["volume"] = simulations[f"volume_{i}"]
                
                simulations_list.append(df)
                del df
            print('Simulations already exist')
        except:
            print("Generating simulation")
            simulator = SimulateMLMertonOHLC(sample, periods)
            if np.isnan(simulator.sigma_jhat):
                simulator = SimulateMLGeometricOHLC(sample, periods)
            simulator.fix_mean_corrfactor(data)
            simulations_list = simulator.simulate_ohlc_n_times_with_variable_mean(np.zeros(num_sim),
                    num_sim, sim_length)
            date_index=pd.date_range(end=sample.index[-1],freq=f"{minutes}t",periods=sim_length)
            df=pd.DataFrame(index=date_index)
            i=0
            for simulations in simulations_list:
                simulations.index=date_index
                df[f"close_{i}"]=simulations["close"]
                df[f"high_{i}"]=simulations["high"]
                df[f"low_{i}"]=simulations["low"]
                df[f"open_{i}"]=simulations["open"]
                df[f"volume_{i}"]=simulations["volume"]
                i+=1
            df.to_csv(f"samples/simulation_{date_final.strftime('%Y%m%d')}-{date_initial.strftime('%Y%m%d')}_{base}{quote}_{minutes}t.csv")
        return simulations_list


#estrategia
class Test(BBvolumeStrategy):
    def get_default_parameters_bounds(self):
        return {"bb_mult_factor":(1.5,4.5),
                #"prop_bb_mult_factor":(0.6,1),
                "bb_periods":(15,100),
                "stop_periods":(15,100),
                "vp_days":(0.51,7.49),
      #          "sign_action":(-1,1)
     #            "vol_lenght":(1*(self.periods/365),7*(self.periods/365)),
      #          "vol_mult_factor_up":(1,3),
      #"vol_mult_factor_down":(1,3),
    #            "pct_candle":(2,50),
#                "stop_loss":(0.99,0.85),
#                "take_profit":(1.05,1.35)
#                
                }
    
    def get_strategy_name(self):
        return "BBvolume_nostop"
    
    def get_actions(self, **kwargs):
        bb_mult_factor = kwargs["bb_mult_factor"]
        #prop_bb_mult_factor = kwargs["prop_bb_mult_factor"]
        
        bb_lenght = int(kwargs["bb_periods"])
        stop_lenght = int(kwargs["stop_periods"])
        vp_days = round(kwargs["vp_days"])
        #sign_action = np.sign(kwargs["sign_action"])
        
#        stop_loss=kwargs["stop_loss"]
#        take_profit=kwargs["take_profit"]
#        
        #vol_lenght = int(kwargs["vol_periods"])
        #vol_mult_factor_up = kwargs["vol_mult_factor_up"]
        #vol_mult_factor_down = kwargs["vol_mult_factor_down"]
        
   #     pct_candle = kwargs["pct_candle"]
       # stop_loss = kwargs["stop_loss"]
        #take_profit = kwargs["take_profit"]
        
        price=self.time_serie    
        #
        price["sma"] = price["close"].rolling(stop_lenght).mean()
        price["sma_bb"] = price["close"].rolling(bb_lenght).mean()
        bbu = price["close"].rolling(bb_lenght).mean() + \
            bb_mult_factor*price["close"].rolling(bb_lenght).std()
        bbl = price["close"].rolling(bb_lenght).mean() - \
            bb_mult_factor*price["close"].rolling(bb_lenght).std()
        kcu = keltner_channel_hband(price["high"],price["low"],price["close"],n=bb_lenght)
        kcl = keltner_channel_lband(price["high"],price["low"],price["close"],n=bb_lenght)
        #dcu = donchian_channel_hband(price["high"],price["low"],price["close"],n=24*dc_days).shift(1)
        #dcl = donchian_channel_lband(price["high"],price["low"],price["close"],n=24*dc_days).shift(1)
        price["bbl"]=bbl
        price["bbh"]=bbu
        
        price["BB"]=(price.close-bbl)/(bbu-bbl)
        #price["DC"]=(price.close-dcl)/(dcu-dcl)
        price["MP"]=(price.close-price[f"value_area_low_{vp_days}"])/(price[f"value_area_high_{vp_days}"]-price[f"value_area_low_{vp_days}"])
        price["KC"]=(price.close-kcl)/(kcu-kcl)
        
        #price["DC"] = np.where(price.index.hour==0,np.nan,price["DC"])
#        #price["DC"] = price["DC"].ffill()
#        bbu = price["close"].rolling(bb_lenght).mean() + \
#            prop_bb_mult_factor*bb_mult_factor*price["close"].rolling(bb_lenght).std()
#        bbl = price["close"].rolling(bb_lenght).mean() - \
#            prop_bb_mult_factor*bb_mult_factor*price["close"].rolling(bb_lenght).std()
#        price["BBnarrow"]=(price.close-bbl)/(bbu-bbl)
#        
        
        
        price["action"] = np.nan
#        price["action"] = np.where(
#                (price["BBnarrow"]>1)&
#                (price["KC"]>1) & 
#                (price["BB"]<1) &
#                (price["DC"]<1) &
#                ((price["BBnarrow"].shift(1)<1)|\
#                 (price["KC"].shift(1)<1))
#                ,-2,price["action"])
#        price["action"] = np.where(
#                (price["BBnarrow"]<0)&
#                (price["KC"]<0) & 
#                (price["BB"]>0) &
#                (price["DC"]>0) &
#                ((price["BBnarrow"].shift(1)>0)|\
#                 (price["KC"].shift(1)>0))
#                , 2,price["action"])
#        
        price["action"] = np.where(
                (price["BB"]>1) &
                (price["KC"]>1) &
                ((price["BB"].shift(1)<1) |
                (price["KC"].shift(1)<1))
                , np.where(price["MP"]>1,1,-2),
                price["action"])
        
        price["action"] = np.where(
                (price["BB"]<0) &
                (price["KC"]<0) &
                ((price["BB"].shift(1)>0) |
                (price["KC"].shift(1)>0)) 
                , np.where(price["MP"]<0,-1,2)
                ,price["action"])

        status = price["action"].ffill()

        price["action"] = np.where(
                (price["MP"]>1) &
                ((price["MP"].shift(1)<1)) &
                (status==-2)
                ,1,
                price["action"])
        
        price["action"] = np.where(
                (price["MP"]<0) &
                ((price["MP"].shift(1)>0)) &
                (status==2)
                ,-1,
                price["action"])
        
        status = price["action"].ffill()

        
        price["action"] = np.where((status==1) &\
             (price["sma"]>=price["close"])
             ,0, price["action"])
        price["action"] = np.where((status==-1) &\
             (price["sma"]<=price["close"])
             ,0, price["action"])
        
        price["action"] = np.where((status == 2) &\
             (price[f"poc_price_{vp_days}"]<=price["close"])
             ,0, price["action"])
        price["action"] = np.where((status == -2) &\
             (price[f"poc_price_{vp_days}"]>=price["close"])
             ,0, price["action"])
        
         
        
        
        
#        
#        action = price.action.values
#        pct = price.close.pct_change().values
#        counting = False
#        short=False
#        cumprod = 1
#        
#        for i in range(len(price)):
#            
#            if action[i] == 1:
#                counting = True
#                short = False
#                cumprod = 1
#            elif action[i] == -1:
#                counting = True
#                short = True
#                cumprod = 1
#            
#            if cumprod>take_profit:
#                counting = False
#                cumprod = 1
#                action[i] = 0
#            elif cumprod<stop_loss:
#                counting = False
#                cumprod = 1
#                action[i]=0
#                
#            if counting:
#                if short:
#                    cumprod *= (-1)*pct[i]
#                    action[i] = -1
#                else:
#                    cumprod *= pct[i]
#                    action[i] = 1
#        price["action"] = action

        price["action"]=price["action"].replace(2,1).replace(-2,-1)
        price["action"]=price["action"].shift(1).ffill().replace(np.nan,0)
        return price["action"]
#funcion para calcular poc y va
def get_poc_max_min(price,n_candles,value_area_pct=0.7):
    poc_array = np.full(n_candles,np.nan)
    min_array = np.full(n_candles,np.nan)
    max_array = np.full(n_candles,np.nan)
    rounded_set = np.round(price["close"],1)
    for i in (range(0,len(price)-n_candles)):    
        profile=price[["open","high","low","close","volume"]].iloc[i:i+n_candles].groupby(rounded_set)['volume'].sum()
        poc_idx=midmax_idx(profile.values)
        target_vol = value_area_pct*profile.sum()
        trial_vol =profile[profile.index[poc_idx]]
        min_idx=poc_idx
        max_idx=poc_idx
        while trial_vol <= target_vol:
                last_min = min_idx
                last_max = max_idx
    
                next_min_idx = np.clip(min_idx - 1, 0, len(profile) - 1)
                next_max_idx = np.clip(max_idx + 1, 0, len(profile) - 1)
    
                low_volume = profile.iloc[next_min_idx] if next_min_idx != last_min else None
                high_volume = profile.iloc[next_max_idx] if next_max_idx != last_max else None
    
                if not high_volume or (low_volume and low_volume > high_volume):
                    trial_vol += low_volume
                    min_idx = next_min_idx
                elif not low_volume or (high_volume and low_volume <= high_volume):
                    trial_vol += high_volume
                    max_idx = next_max_idx
                else:
                    break
        poc_array = np.append(poc_array,profile.index[poc_idx])
        min_array = np.append(min_array,profile.index[min_idx])
        max_array = np.append(max_array,profile.index[max_idx])
    
    return poc_array,min_array,max_array
#funcion para guardar simulaciones con poc y va
def save_sim(simulations_list,base,quote,date_final,minutes=60):
    df=pd.DataFrame(index=simulations_list[0].index)
    i=0
    
    #glob(f"samples/simulation_{date.strftime('%Y%m%d')}*"_{base}{quote}_{minutes}t.csv")
    
    for simulations in simulations_list:
        df[f"close_{i}"]=simulations["close"]
        df[f"high_{i}"]=simulations["high"]
        df[f"low_{i}"]=simulations["low"]
        df[f"open_{i}"]=simulations["open"]
        df[f"volume_{i}"]=simulations["volume"]
        df[f"poc_price_6_{i}"]=simulations["poc_price_6"]
        df[f"value_area_low_6_{i}"]=simulations["value_area_low_6"]
        df[f"value_area_high_6_{i}"]=simulations["value_area_high_6"]
        i+=1
    df.to_csv(f"samples/simulation_{date_final.strftime('%Y%m%d')}-{date_initial.strftime('%Y%m%d')}_{base}{quote}_{minutes}t.csv")
#leo datos histórico y calculo poc y va
bases=["ETH"]
quotes=["BTC","USDT"]
mult_factor={"USDT":1,"BTC":10000}
dict_price={"ETH":{}} 
for base,quote in itertools.product(bases,quotes):
    dict_price[base][quote] = repo.get_candlestick(
                f"{base}/{quote}", 
                "binance", 
                60,
                datetime(2021,1,1), 
                datetime(2022,1,1))
    dict_price[base][quote]["open"] = mult_factor[quote]*dict_price[base][quote]["open"]
    dict_price[base][quote]["low"] = mult_factor[quote]*dict_price[base][quote]["low"]
    dict_price[base][quote]["high"] = mult_factor[quote]*dict_price[base][quote]["high"]
    dict_price[base][quote]["close"] = mult_factor[quote]*dict_price[base][quote]["close"]
    price=dict_price[base][quote].copy()
    poc_price,value_area_low,value_area_high = get_poc_max_min(price,24*6)
    dict_price[base][quote][f"poc_price_6"]=poc_price
    dict_price[base][quote][f"value_area_low_6"]=value_area_low
    dict_price[base][quote][f"value_area_high_6"]=value_area_high
# genero simulaciones
simulations={"ETH":{"USDT":{},"BTC":{}}}
for base,quote,date in itertools.product(bases,quotes,dates):
    input_data = FillNa().fill_ohlc(dict_price[base][quote].loc[:date])
    simulations[base][quote][date]=generate_simulation(input_data,60,base,quote,num_sim=100,sim_length=3000)
#calculo de poc  va paralelizado
ray.init()
@ray.remote
def calc_poc_and_va(simulation):
    price=simulation.copy()
    def get_poc_max_min(price,n_candles,value_area_pct=0.7):
        poc_array = np.full(n_candles,np.nan)
        min_array = np.full(n_candles,np.nan)
        max_array = np.full(n_candles,np.nan)
        rounded_set = np.round(price["close"],1)
        for i in (range(0,len(price)-n_candles)):    
            profile=price[["open","high","low","close","volume"]].iloc[i:i+n_candles].groupby(rounded_set)['volume'].sum()
            poc_idx=midmax_idx(profile.values)
            target_vol = value_area_pct*profile.sum()
            trial_vol =profile[profile.index[poc_idx]]
            min_idx=poc_idx
            max_idx=poc_idx
            while trial_vol <= target_vol:
                    last_min = min_idx
                    last_max = max_idx
        
                    next_min_idx = np.clip(min_idx - 1, 0, len(profile) - 1)
                    next_max_idx = np.clip(max_idx + 1, 0, len(profile) - 1)
        
                    low_volume = profile.iloc[next_min_idx] if next_min_idx != last_min else None
                    high_volume = profile.iloc[next_max_idx] if next_max_idx != last_max else None
        
                    if not high_volume or (low_volume and low_volume > high_volume):
                        trial_vol += low_volume
                        min_idx = next_min_idx
                    elif not low_volume or (high_volume and low_volume <= high_volume):
                        trial_vol += high_volume
                        max_idx = next_max_idx
                    else:
                        break
            poc_array = np.append(poc_array,profile.index[poc_idx])
            min_array = np.append(min_array,profile.index[min_idx])
            max_array = np.append(max_array,profile.index[max_idx])
        
        return poc_array,min_array,max_array

    poc_price,value_area_low,value_area_high = get_poc_max_min(price,24*6)
    simulation[f"poc_price_6"]=poc_price
    simulation[f"value_area_low_6"]=value_area_low
    simulation[f"value_area_high_6"]=value_area_high
    return simulation
print(100*2*len(dates))
#calculo poc y va de las simulaciones
simulations2=simulations.copy()    
for sample_id in tqdm(itertools.product(bases,quotes,dates)):
    futures = [calc_poc_and_va.remote(simulations[base][quote][date][sample_id]) for sample_id in range(100)]
    simulations2[base][quote][date] = ray.get(futures)
#guardo simulaciones con poc y va calculados
for sample_id,base,quote,date in itertools.product(range(100),bases,quotes,dates):
    save_sim(simulations2[base][quote][date],base,quote,date,minutes=60)

#Genero parámetros random de muestras
strategy_hist={"ETH":{}}
for quote in quotes:
    strategy_hist["ETH"][quote]=Test([dict_price["ETH"][quote]],
                         "ETH",
                         quote,
                         {"USDT":0.05e-2,
                          "BTC":0.02e-2,
                          "ETH":0.0275e-2
                          },
                          date_init=price.loc[datetime(2020,7,1)].name,
                          date_final=price.loc[datetime(2020,9,1)].name,
                          candle_minutes=60,
                          optimize="sharpe"
                          )    

    strategy_hist["ETH"][quote].pbounds["vp_days"]=(6,6)
random_parameters=[]
for i in range(0,2000):
    random_parameters.append({item[0]:np.random.uniform(low=item[1][0],high=item[1][1],size=1)[0]  for item in strategy_hist["ETH"]["USDT"].pbounds.items()})
df_rp=pd.DataFrame(random_parameters)
df_rp["bb_periods"]=df_rp["bb_periods"].astype(int)
df_rp["stop_periods"]=df_rp["stop_periods"].astype(int)
df_rp["vp_days"]=df_rp["vp_days"].astype(int)
df_rp["bb_mult_factor"]=df_rp["bb_mult_factor"].apply(lambda x : round(x,2))
df_rp=df_rp.drop_duplicates()

#función de calculo de métricas
@ray.remote
def calc_metrics(strategy,date_init_train,date_final_train,date_final_test):
    result=[]
    for idx,row in (df_rp.iterrows()):
        params = row.to_dict()
        strategy.set_dates(date_init_train,date_final_train)
        returns1 = strategy.get_returns(0,**params)
        actions = strategy.get_actions(**params).loc[returns1.index]
        #actions = strategy.get_actions(**point["params"])
        if (returns1==0).all():
            continue
        kelly=qs.stats.kelly_criterion(returns1)
        volatility=qs.stats.volatility(returns1,periods=365*24)
        cum_ret=(1+returns1).cumprod()
        returns=cum_ret[-1]
        mean_return=returns1.mean()*365*24
        keep_trades = ((actions.shift(-1)==1) & (actions==0)) |\
                    ((actions.shift(-1)==1) & (actions==-1))|\
                    (np.isnan(actions.shift(-1)) & (actions==0))|\
                    (np.isnan(actions.shift(-1)) & (actions==-1))
        
        returns_per_trade = cum_ret.loc[(keep_trades)].pct_change().dropna()
        pos_trades=len(returns_per_trade[returns_per_trade>0])
        neg_trades=len(returns_per_trade[returns_per_trade<0])
        avg_pos=(returns_per_trade[returns_per_trade>0]).mean()
        avg_neg=(returns_per_trade[returns_per_trade<0]).mean()
        max_DD = qs.stats.max_drawdown(returns1)
        #strategy[base].set_dates(price.loc[datetime(2020,7,10)].name,price.index[-1])  
        strategy.set_dates(date_final_train,date_final_test)  
        returns1 = strategy.get_returns(0,**params)
        future_returns=(1+returns1).cumprod()[-1]
        
        result.append({
                "id":idx,
                "date_init_train":date_init_train,
                "date_final_train":date_final_train,
                "date_final_test":date_final_test,
                "params":params,
                "kelly":kelly,
                "volatility":volatility,
                "returns":returns,
                "mean_return":mean_return,
                "pos_trades":pos_trades,
                "neg_trades":neg_trades,
                "avg_pos":avg_pos,
                "avg_neg":avg_neg,
                "max_DD":max_DD,
                "future": future_returns
                })
    return result

##calculo las métricas para todos los parámetros generados
result={}
for base in bases:
    result={base:{}}
    for quote in quotes:
        strategy=Test([dict_price[base][quote]],
                      base,
                      quote,
                      {"USDT":0.05e-2,
                       "BTC":0.02e-2,
                       "ETH":0.0275e-2
                       },
                       date_init=price.loc[datetime(2020,1,1)].name,
                       date_final=price.loc[datetime(2020,9,1)].name,
                       candle_minutes=60,
                       optimize="sharpe"
                      )
        date_to_metrics=[]
        for date in dates:
            date_init_train=date-timedelta(weeks=30)
            date_final_train=date
            date_final_test=date+timedelta(weeks=1)
            date_to_metrics.append({"date_init_train":date_init_train,
                                    "date_final_train":date_final_train,
                                    "date_final_test":date_final_test,
                                    })
        metrics = [calc_metrics(strategy,**date) for date in tqdm(date_to_metrics)]
        result[base][quote]=metrics
# guardo las métricas
for base,quote in bases,quotes:
    for i in range(len(result[base][quote])):
        df=pd.DataFrame(result[quote][quote][i])
        df.to_csv(f"metrics_to_optimize/metrics_{base}{quote}_{df['date_final_train'].iloc[0].strftime('%Y-%m-%d')}.csv")
    
from xgboost import XGBRegressor
#entreno un regresor para obtener los mejores parámetros
cla={}
for base in bases:
    cla[base]={}
    for quote in quotes:
        cla[base][quote]={}
        for date in dates:
            df=pd.read_csv(f"metrics_to_optimize/metrics_{base}{quote}_{date.strftime('%Y-%m-%d')}.csv")
            X=df[["mean_return","volatility","max_DD","neg_trades","pos_trades","avg_neg","avg_pos"]]    
            y=df["future"]
            cla[base][quote][date] = XGBRegressor()
            cla[base][quote][date].fit(X,y)
#utilizo el regresor en la semana siguiente
future_returns={"ETH":{"BTC":pd.DataFrame(),"USDT":pd.DataFrame()}}
for base,quote in itertools.product(bases,quotes):
    for date in dates[1:]:
        df=pd.read_csv(f"metrics_to_optimize/metrics_{base}{quote}_{date.strftime('%Y-%m-%d')}.csv")
        df["params"]=df["params"].apply(lambda params: json.loads(params.replace("'",'"')))

        X=df[["mean_return","volatility","max_DD","neg_trades","pos_trades","avg_neg","avg_pos"]]    
        y=df["future"]
        y_pred=cla[base][quote][date-timedelta(weeks=1)].predict(X)
        idx_max = y_pred.argmax()
        future_returns[base][quote] = future_returns[base][quote].append(df.iloc[idx_max])
    future_returns[base][quote] = future_returns[base][quote].drop(columns="Unnamed: 0")

#Realizo el backtest con los parámetros calculados
df = pd.DataFrame()
for base,quote in itertools.product(bases,quotes):
    strategy=Test([dict_price[base][quote]],
              base,
              quote,
              {"USDT":0.05e-2,
               "BTC":0.02e-2,
               "ETH":0.0275e-2
               },
               date_init=price.loc[datetime(2020,1,1)].name,
               date_final=price.loc[datetime(2020,9,1)].name,
               candle_minutes=60,
               optimize="sharpe"
              )
    returns = pd.Series()
    returns_curr = pd.Series()
    for i, row in future_returns[base][quote].iterrows():
        strategy.set_dates(datetime.strptime(row["date_final_train"],"%Y-%m-%d"),
                           datetime.strptime(row["date_final_test"],"%Y-%m-%d"))
        returns_temp = strategy.get_returns(0,**row["params"])
        returns = returns.append(returns_temp)
        returns_curr = returns_curr.append(dict_price[base][quote]["close"].pct_change().loc[row["date_final_train"]:row["date_final_test"]])
    df[f"strategy_{base}{quote}"] = returns.loc[~returns.index.duplicated("last")]
    returns_curr = returns_curr.loc[returns.index]
    df[f"{base}{quote}"] = returns_curr.loc[~returns_curr.index.duplicated("last")]

#Gráfico
df.to_csv("strategy_over_ETH.csv")
(1+df).cumprod().plot()
