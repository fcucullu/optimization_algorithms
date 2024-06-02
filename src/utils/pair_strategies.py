import pandas as pd
import numpy as np
from decimal import Decimal as D
from ta.volatility import bollinger_lband,bollinger_hband, keltner_channel_lband,keltner_channel_hband
from ta.momentum import rsi    
from ta.trend import adx, adx_neg, adx_pos, macd_diff
from utils.candlestick import CandlestickRepository, MINUTES_TO_BINANCE_INTERVAL
from market_profile import MarketProfile
BinancePricesRepository = CandlestickRepository.default_repository

class Holding:
    def __init__(self, base_currency: str, quote_currency: str, time_frame: int, prices_repository=BinancePricesRepository()):
        self.base_currency = base_currency
        self.quote_currency = quote_currency
        self.time_frame = int(time_frame)
        self.prices_repository = prices_repository

    def run(self, last_recommendation=None):
        candles_data = self.get_candles_data()
        open_prices = candles_data.open         
        
        proportion_to_hold=1
        timestamp = open_prices.index[-1]

        return proportion_to_hold, timestamp, self.calculate_prices_variation(open_prices)

    def get_candles_data(self):
        """ get the candles data needed to process the strategy """
        n_candles = self.get_candles_needed_for_running()
        candles = self.prices_repository.get_candles(
            self.base_currency, 
            self.quote_currency, 
            MINUTES_TO_BINANCE_INTERVAL[self.time_frame], 
            n_candles)

        return candles
    
    def get_open_prices(self):
        candles_data = self.get_candles_data()
        return candles_data.open

    def get_candles_needed_for_running(self):
        """ Returns the numbers of candles needed to process the strategy """
        return 2
    
    def calculate_prices_variation(self, open_prices):
        price_variation = (open_prices[-1] / open_prices[-2])
        return D(price_variation)


class OutOfMarket(Holding):
    def run(self, last_recommendation):
        candles_data = self.get_candles_data()
        open_prices = candles_data.open         
        
        proportion_to_hold=0
        timestamp = open_prices.index[-1]

        return proportion_to_hold, timestamp, self.calculate_prices_variation(open_prices)


class TwoStandardMovingAverage(Holding):

    def __init__(self, base_currency: str, quote_currency: str,
                 time_frame: int,
                 prices_repository=BinancePricesRepository(),**kwargs):
        super().__init__(base_currency, quote_currency, time_frame, prices_repository)
        self.ma_long = int(kwargs["ma_long"])
        self.ma_short = int(kwargs["ma_short"])

    def run(self, **kwargs):
        """ 
        return strategy status. Value 1 means hold, 0 means do not hold. 
        """
        candles_data = self.get_candles_data()
        open_prices = candles_data.open 
        long = open_prices[-self.ma_long:].mean()
        short = open_prices[-self.ma_short:].mean()
        
        proportion_to_hold = 1 if long < short else 0
        timestamp = open_prices.index[-1]

        return proportion_to_hold, timestamp, self.calculate_prices_variation(open_prices)

    def get_candles_needed_for_running(self):
        """ Returns the numbers of candles needed to process the strategy """
        return self.ma_long
        

class ThreeStandardMovingAverage(Holding):

    def __init__(self,
        base_currency: str, 
        quote_currency: str, 
        time_frame: int, 
        prices_repository=BinancePricesRepository(),
        **kwargs):
        
        super().__init__(base_currency, quote_currency, time_frame, prices_repository)
        self.ma_long = int(kwargs["ma_long"])
        self.ma_short_buy = int(kwargs["ma_short_buy"])
        self.ma_short_sell = int(kwargs["ma_short_sell"])

    def run(self, last_recommendation: float = 0, **kwargs):
        """ 
        return strategy status
        """
        open_prices = self.get_open_prices()
        long = open_prices[-self.ma_long:].mean()
        short_buy = open_prices[-self.ma_short_buy:].mean()
        short_sell = open_prices[-self.ma_short_sell:].mean()
        
        timestamp = open_prices.index[-1]

        proportion_to_hold = last_recommendation    
        #Pongo primero la regla de estar out-of-market para que importe mas en la decision
        if long>short_sell:
            proportion_to_hold = 0
        elif long<short_buy:
            proportion_to_hold =  1
        
        
        return proportion_to_hold, timestamp, self.calculate_prices_variation(open_prices)

    def get_candles_needed_for_running(self):
        """ Returns the numbers of candles needed to process the strategy """
        return self.ma_long


class ThreeStandardMovingAverageAlternative(Holding):

    def __init__(self, 
        base_currency: str, quote_currency: str, time_frame: int, 
        ma_long: int, ma_medium: int, ma_short: int,
        prices_repository=BinancePricesRepository(),**kwargs):

        super().__init__(base_currency, quote_currency, time_frame, prices_repository)
        self.ma_long = int(ma_long)
        self.ma_medium = int(ma_medium)
        self.ma_short = int(ma_short)

    def run(self, last_recommendation: float = 0, **kwargs):
        """ 
        return strategy status
        """
        open_prices = self.get_open_prices()
        long = open_prices[-self.ma_long:].mean()
        medium = open_prices[-self.ma_medium:].mean()
        short = open_prices[-self.ma_short:].mean()
        
        timestamp = open_prices.index[-1]
        
        proportion_to_hold = last_recommendation
        if long < short and medium < short:
            proportion_to_hold = 1 
        elif long > short and medium > short:
            proportion_to_hold = 0
        
        return proportion_to_hold, timestamp, self.calculate_prices_variation(open_prices)

    def get_candles_needed_for_running(self):
        """ Returns the numbers of candles needed to process the strategy """
        return self.ma_long


class VWAPvsSMA(Holding):
    def __init__(self, 
        base_currency: str, quote_currency: str, time_frame: int,
        n_candles:int, min_percentage:float,     
        prices_repository=BinancePricesRepository()):
        
        super().__init__(base_currency, quote_currency, time_frame, prices_repository)        
        self.n_candles = int(n_candles) 
        self.min_percentage = min_percentage
         
    def run(self, last_recommendation: float = 0, **kwargs):
        candles_data = self.get_candles_data()
        open_prices = candles_data.open
        volume = candles_data.volume 

        vwap = self.calculate_vwap(open_prices, volume)
        sma = open_prices[-self.n_candles:].mean()
        current_percentage = abs(np.log(vwap/sma))

        timestamp = open_prices.index[-1]

        recommendation = last_recommendation
        if current_percentage >= self.min_percentage:
            if vwap > sma:
                recommendation = 1
            else:
                recommendation = 0

        return recommendation, timestamp, self.calculate_prices_variation(open_prices)

    def calculate_vwap(self, open_prices, volume): 
        volume_times_price_mean = (volume[-self.n_candles+1:-1] * open_prices[-self.n_candles:]).mean()
        volume_mean = volume[-self.n_candles+1:-1].mean()
        return volume_times_price_mean / volume_mean

    def get_candles_needed_for_running(self):
        return self.n_candles + 1

class ADXStrategy(Holding):

    rsi_periods = 14
    constant_stoch_rsi = 100
    max_number_of_candles = 1000

    def __init__(self, 
        base_currency: str, 
        quote_currency: str,
        time_frame: int, 
        bb_periods: int, 
        kc_periods: int,
        std_bb: float, 
        std_kc: float, 
        stoch_rsi_periods: int, 
        adx_periods: int, 
        stoch_rsi_upper: float, 
        stoch_rsi_low: float, 
        adx_neg_bound: float, 
        adx_bound: float, 
        prices_repository=BinancePricesRepository()):

        super().__init__(base_currency, quote_currency, time_frame, prices_repository)         
        self.bb_periods = int(bb_periods)
        self.kc_periods = int(kc_periods)
        self.std_bb = std_bb
        self.std_kc = std_kc
        self.stoch_rsi_periods = int(stoch_rsi_periods)
        self.adx_periods = int(adx_periods)
        self.stoch_rsi_upper = stoch_rsi_upper
        self.stoch_rsi_low = stoch_rsi_low
        self.adx_neg_bound = adx_neg_bound
        self.adx_bound = adx_bound

    def run(self, last_recommendation: float = 0, **kwargs):
        """ 
        return strategy status. Value 1 means hold, 0 means do not hold. 
        """
        candles_data = self.get_candles_data()
        open_prices = candles_data.open
        high_prices = candles_data.high
        low_prices = candles_data.low
   
        current_variables = self.get_value_of_current_variables(open_prices, high_prices.shift(1), low_prices.shift(1))
        current_adx = current_variables[0] 
        previous_adx = current_variables[1]
        current_adx_neg = current_variables[2] 
        current_adx_pos = current_variables[3]
        current_bb_lband = current_variables[4] 
        current_kc_lband = current_variables[5]
        current_stoch_rsi = current_variables[6]

        timestamp = open_prices.index[-1]
        
        hold = current_stoch_rsi > self.stoch_rsi_upper \
                and current_adx > previous_adx \
                and current_adx_pos > current_adx_neg 

        not_hold_lbands = open_prices[-1] < current_bb_lband \
                or open_prices[-1] < current_kc_lband
        not_hold = current_stoch_rsi < self.stoch_rsi_low \
                and current_adx_neg > self.adx_neg_bound \
                and current_adx > self.adx_bound \
                and current_adx_neg > current_adx_pos \

        if hold:
            recommendation = 1
        elif not_hold_lbands and not_hold:
            recommendation = 0
        else:
            recommendation = last_recommendation
        
        return recommendation, timestamp, self.calculate_prices_variation(open_prices)

    def calculate_stoch_rsi(self,data: pd.Series, period: int):
        series_rsi = rsi(data,window = self.rsi_periods, fillna=True)
        series_rsi_min = series_rsi.rolling(window=period).min()
        series_rsi_max = series_rsi.rolling(window=period).max()
        series_stoch_rsi = self.constant_stoch_rsi * (series_rsi - series_rsi_min) / (series_rsi_max - series_rsi_min)    
        return series_stoch_rsi

    def get_value_of_current_variables(self, open_prices, high_prices, low_prices):
        adx_data = adx(high_prices, low_prices, open_prices, window=self.adx_periods)[-2:]
        current_adx, previous_adx = adx_data[1], adx_data[0]
        current_adx_neg = adx_neg(high_prices, low_prices, open_prices, window=self.adx_periods)[-1] 
        current_adx_pos = adx_pos(high_prices, low_prices, open_prices, window=self.adx_periods)[-1]
        current_bb_lband = bollinger_lband(open_prices, window=self.bb_periods, window_dev=self.std_bb)[-1]
        current_kc_lband = keltner_channel_lband(high_prices, low_prices, open_prices, window=self.kc_periods, window_atr=self.kc_periods)[-1]
        current_stoch_rsi = self.calculate_stoch_rsi(open_prices, self.stoch_rsi_periods)[-1] 
        return current_adx, previous_adx, current_adx_neg, current_adx_pos, current_bb_lband, current_kc_lband, current_stoch_rsi    
       
    def get_candles_needed_for_running(self):
        return self.max_number_of_candles #mientras más velas mejor, para el cálculo de algunos indicadores

#########################################################################

class ADXPPStrategy(ADXStrategy):
    
    def run(self, last_recommendation: float = 0, **kwargs):
        """ 
        return strategy status. Value 1 means hold, 0 means do not hold. 
        """
        candles_data = self.get_candles_data()
        candles_data = self.increase_values_to_significant_magnitudes(candles_data, self.quote_currency)
        
        open_prices = candles_data.open
        high_prices = candles_data.high
        low_prices = candles_data.low
        
        current_variables = self.get_value_of_current_variables(open_prices, high_prices.shift(1), low_prices.shift(1))
        current_adx = current_variables[0] 
        previous_adx = current_variables[1]
        current_adx_neg = current_variables[2] 
        current_adx_pos = current_variables[3]
        current_bb_lband = current_variables[4] 
        current_kc_lband = current_variables[5]
        current_stoch_rsi = current_variables[6]

        one_day_data = self.get_one_day_data(open_prices, high_prices.shift(1), low_prices.shift(1))
        current_support = self.get_support(one_day_data.open, one_day_data.high, one_day_data.low)

        timestamp = open_prices.index[-1]
        
        hold = current_stoch_rsi > self.stoch_rsi_upper \
            and current_adx > previous_adx \
            and current_adx_pos > current_adx_neg 

        not_hold_lbands = open_prices[-1] < current_bb_lband \
                or open_prices[-1] < current_kc_lband

        not_hold = current_stoch_rsi < self.stoch_rsi_low \
                and current_adx_neg > self.adx_neg_bound \
                and current_adx > self.adx_bound \
                and current_adx_neg > current_adx_pos 

        not_hold_support = open_prices[-1] < current_bb_lband \
                        and low_prices.shift(1)[-1] < current_support

        if hold:
            recommendation = 1
        elif (not_hold_lbands and not_hold) or not_hold_support:
            recommendation = 0
        else:
            recommendation = last_recommendation

        return recommendation, timestamp, self.calculate_prices_variation(open_prices)

    def increase_values_to_significant_magnitudes(self, candles_data, quote):
        """
        If quote_currency is BTC, the prices are too small, so we need to multiplier for 1e3 in order to avoid 
        numerical errors. 
        """
        if quote == 'BTC':
            numeric_columns = candles_data._get_numeric_data().columns
            candles_data[numeric_columns] = 1000*candles_data[numeric_columns]
        return candles_data            

    def get_one_day_data(self,open_prices, high_prices, low_prices):
        index_1d = pd.date_range(start=open_prices.index[0], end=open_prices.index[-1], freq='1d')
        df_1d = pd.DataFrame(columns= ['open', 'high', 'low'], index=index_1d)
        df_1d['high'] = open_prices.resample('1D').max()
        df_1d['low'] = low_prices.resample('1D').min()
        df_1d['open'] = open_prices[df_1d.index]
        df_1d = df_1d.fillna(method='ffill') 
        return df_1d
            
    def get_support(self,open_prices, high_prices, low_prices):
        """
        Get support 1 for one day data. 
        """
        pp = (high_prices + low_prices + open_prices)/3
        support = 2 * pp - high_prices
        support = support.shift(1)
        return support[-1]

#########################################################################

class SignalForceStrategy(Holding):
    def __init__(self,
            base_currency: str, quote_currency: str, time_frame: int,
            macd_requp: float,
            macd_reqdown: float,
            macd_signalup: float,
            macd_signaldown: float,
            macd_diff_signalup: float,
            macd_diff_signaldown: float,
            vol_requp: float,
            vol_reqdown: float,
            vol_signalup: float,
            vol_signaldown: float,
            vol_diff_signalup: float,
            vol_diff_signaldown: float,
            bb_requp: float,
            bb_reqdown: float,
            bb_signalup: float,
            bb_signaldown: float,
            bb_diff_signalup: float,
            bb_diff_signaldown: float,
            rsi_requp: float,
            rsi_reqdown: float,
            rsi_signalup: float,
            rsi_signaldown: float,
            rsi_diff_signalup: float,
            rsi_diff_signaldown: float,
            prices_repository=BinancePricesRepository()):
        super().__init__(base_currency, quote_currency, time_frame, prices_repository)         
        self.macd_requp = macd_requp
        self.macd_reqdown = macd_reqdown 
        self.macd_signalup = macd_signalup 
        self.macd_signaldown = macd_signaldown 
        self.macd_diff_signalup = macd_diff_signalup 
        self.macd_diff_signaldown = macd_diff_signaldown 
        self.vol_requp = vol_requp 
        self.vol_reqdown = vol_reqdown 
        self.vol_signalup = vol_signalup 
        self.vol_signaldown = vol_signaldown 
        self.vol_diff_signalup = vol_diff_signalup 
        self.vol_diff_signaldown = vol_diff_signaldown 
        self.bb_requp = bb_requp 
        self.bb_reqdown = bb_reqdown 
        self.bb_signalup = bb_signalup 
        self.bb_signaldown = bb_signaldown 
        self.bb_diff_signalup = bb_diff_signalup 
        self.bb_diff_signaldown = bb_diff_signaldown 
        self.rsi_requp = rsi_requp 
        self.rsi_reqdown = rsi_reqdown 
        self.rsi_signalup = rsi_signalup 
        self.rsi_signaldown = rsi_signaldown 
        self.rsi_diff_signalup = rsi_diff_signalup 
        self.rsi_diff_signaldown = rsi_diff_signaldown  
        self.macd_diff_requp,self.macd_diff_reqdown = 0.,0.
        self.vol_diff_requp,self.vol_diff_reqdown = 0.,0.
        self.bb_diff_requp,self.bb_diff_reqdown = 0.,0.
        self.rsi_diff_requp,self.rsi_diff_reqdown = 0.,0.
        
          
    def run(self,last_recommendation: float = 0, **kwargs):
        """ 
        return strategy status. Value 1 means hold, 0 means do not hold. 
        """
        candles_data = self.get_candles_data()
        open_prices = candles_data.open
        candles_data = candles_data.shift(1)
        volume = candles_data.volume
        timestamp = open_prices.index[-1]
        
        MACD = macd_diff(open_prices,fillna=True).values
        BBH = bollinger_hband(open_prices,fillna=True).values
        BBL = bollinger_lband(open_prices,fillna=True).values
        RSI = rsi(open_prices,fillna=True).values
        VWAP=((open_prices*volume).rolling(30,min_periods=0).mean()/volume.rolling(30,min_periods=0).mean()).values
        SMA=(open_prices).rolling(30,min_periods=0).mean().values
        BB=(open_prices-BBL)/(BBH-BBL)
        VOL=10000.*np.log(VWAP/SMA)
        
        VOL_diff = (VOL[-1]-VOL[-2])/VOL[-2]
        MACD_diff = (MACD[-1]-MACD[-2])/MACD[-2]
        BB_diff = (BB[-1]-BB[-2])/BB[-2]
        RSI_diff = (RSI[-1]-RSI[-2])/RSI[-2]
        
        VOL = VOL[-1]
        MACD = MACD[-1]
        BB = BB[-1]
        RSI = RSI[-1]
        
        signal=0.
        #requiments over technical indicators
        
        if MACD>self.macd_requp:
            signal+=self.macd_signalup
        elif MACD<self.macd_reqdown: 
            signal+=self.macd_signaldown
        
        if VOL>self.vol_requp:
            signal+=self.vol_signalup
        elif VOL<self.vol_reqdown:
            signal+=self.vol_signaldown
        
        if RSI>self.rsi_requp:
            signal+=self.rsi_signalup
        elif RSI<self.rsi_reqdown: 
            signal+=self.rsi_signaldown
        
        if BB>self.bb_requp:
            signal+=self.bb_signalup
        elif BB<self.bb_reqdown: 
            signal+=self.bb_signaldown
        
        if MACD_diff>0:
            signal+=self.macd_diff_signalup
        elif MACD_diff<0: 
            signal+=self.macd_diff_signaldown
        
        if VOL_diff>0:
            signal+=self.vol_diff_signalup
        elif VOL_diff<0: 
            signal+=self.vol_diff_signaldown
        
        if RSI_diff>0:
            signal+=self.rsi_diff_signalup
        elif RSI_diff<0: 
            signal+=self.rsi_diff_signaldown
        
        if BB_diff>0:
            signal+=self.bb_diff_signalup
        elif BB_diff<0: 
            signal+=self.bb_diff_signaldown
        
        #signal requirements
        if signal>100.:
            proportion_to_hold = 1.
        elif signal<-100.:
            proportion_to_hold = 0.
        else :
            proportion_to_hold = last_recommendation
        return proportion_to_hold, timestamp, self.calculate_prices_variation(open_prices)

    def get_candles_needed_for_running(self):
        """ Returns the numbers of candles needed to process the strategy """
        return 100

#########################################################################

class SignalForceAltStrategy(Holding):
    
    def __init__(self, base_currency: str, quote_currency: str, time_frame: int,
                 macd_requp:float,
                 macd_reqdown:float,
                 macd_signalupup:float,
                 macd_signalupdown:float,
                 macd_signaldownup:float,
                 macd_signaldowndown:float,
                 vol_requp:float,
                 vol_reqdown:float,
                 vol_signalupup:float,
                 vol_signalupdown:float,
                 vol_signaldownup:float,
                 vol_signaldowndown:float,
                 bb_requp:float,
                 bb_reqdown:float,
                 bb_signalupup:float,
                 bb_signalupdown:float,
                 bb_signaldownup:float,
                 bb_signaldowndown:float,
                 rsi_requp:float,
                 rsi_reqdown:float,
                 rsi_signalupup:float,
                 rsi_signalupdown:float,
                 rsi_signaldownup:float,
                 rsi_signaldowndown:float,
                 prices_repository=BinancePricesRepository()):
        super().__init__(base_currency, quote_currency, time_frame, prices_repository)
        self.macd_requp =         macd_requp         
        self.macd_reqdown =       macd_reqdown      
        self.macd_signalupup =    macd_signalupup   
        self.macd_signalupdown =  macd_signalupdown 
        self.macd_signaldownup =  macd_signaldownup 
        self.macd_signaldowndown =macd_signaldowndown
        self.vol_requp =          vol_requp         
        self.vol_reqdown =        vol_reqdown       
        self.vol_signalupup =     vol_signalupup    
        self.vol_signalupdown =   vol_signalupdown  
        self.vol_signaldownup =   vol_signaldownup  
        self.vol_signaldowndown = vol_signaldowndown
        self.bb_requp =           bb_requp          
        self.bb_reqdown =         bb_reqdown        
        self.bb_signalupup =      bb_signalupup     
        self.bb_signalupdown =    bb_signalupdown   
        self.bb_signaldownup =    bb_signaldownup   
        self.bb_signaldowndown =  bb_signaldowndown 
        self.rsi_requp =          rsi_requp         
        self.rsi_reqdown =        rsi_reqdown       
        self.rsi_signalupup =     rsi_signalupup    
        self.rsi_signalupdown =   rsi_signalupdown  
        self.rsi_signaldownup =   rsi_signaldownup  
        self.rsi_signaldowndown = rsi_signaldowndown
        self.macd_diff_requp,self.macd_diff_reqdown=0.,0.
        self.vol_diff_requp,self.vol_diff_reqdown=0.,0.
        self.bb_diff_requp,self.bb_diff_reqdown=0.,0.
        self.rsi_diff_requp,self.rsi_diff_reqdown=0.,0.
        
        
    def run(self,last_recommendation: float = 0, **kwargs):
        """ 
        return strategy status. Value 1 means hold, 0 means do not hold. 
        """
        candles_data = self.get_candles_data()
        open_prices = candles_data.open
        volume = candles_data.shift(1).volume
        timestamp = open_prices.index[-1]
        
        MACD = macd_diff(open_prices,fillna=True).values
        BBH = bollinger_hband(open_prices,fillna=True).values
        BBL = bollinger_lband(open_prices,fillna=True).values
        RSI = rsi(open_prices,fillna=True).values
        VWAP=((open_prices*volume).rolling(30,min_periods=0).mean()/volume.rolling(30,min_periods=0).mean()).values
        SMA=(open_prices).rolling(30,min_periods=0).mean().values
        BB=(open_prices-BBL)/(BBH-BBL)
        VOL=10000.*np.log(VWAP/SMA)
        
        VOL_diff = (VOL[-1]-VOL[-2])/VOL[-2]
        MACD_diff = (MACD[-1]-MACD[-2])/MACD[-2]
        BB_diff = (BB[-1]-BB[-2])/BB[-2]
        RSI_diff = (RSI[-1]-RSI[-2])/RSI[-2]
        VOL = VOL[-1]
        MACD = MACD[-1]
        BB = BB[-1]
        RSI = RSI[-1]
        signal=0.

        #requiments over technical indicators
        if (MACD>self.macd_requp) & (MACD_diff>self.macd_diff_requp):
            signal += self.macd_signalupup
        if (MACD>self.macd_requp) & (MACD_diff<self.macd_diff_reqdown):
            signal += self.macd_signalupdown
        if( MACD<self.macd_reqdown) & (MACD_diff>self.macd_diff_requp) :
            signal += self.macd_signaldownup
        if (MACD<self.macd_reqdown) & (MACD_diff<self.macd_diff_reqdown) :
            signal += self.macd_signaldowndown
        
        if (VOL>self.vol_requp) & (VOL_diff>self.vol_diff_requp) :
            signal += self.vol_signalupup
        if (VOL>self.vol_requp) & (VOL_diff<self.vol_diff_reqdown) :
            signal += self.vol_signalupdown
        if (VOL<self.vol_reqdown) & (VOL_diff>self.vol_diff_requp) :
            signal += self.vol_signaldownup
        if (VOL<self.vol_reqdown) & (VOL_diff<self.vol_diff_reqdown) :
            signal += self.vol_signaldowndown
        
        if (BB>self.bb_requp) & (BB_diff>self.bb_diff_requp) :
            signal += self.bb_signalupup
        if (BB>self.bb_requp) & (BB_diff<self.bb_diff_reqdown) :
            signal += self.bb_signalupdown
        if (BB<self.bb_reqdown) & (BB_diff>self.bb_diff_requp) :
            signal += self.bb_signaldownup
        if (BB<self.bb_reqdown) & (BB_diff<self.bb_diff_reqdown) :
            signal += self.bb_signaldowndown
        
        if (RSI>self.rsi_requp) & (RSI_diff>self.rsi_diff_requp) :
            signal += self.rsi_signalupup
        if (RSI>self.rsi_requp) & (RSI_diff<self.rsi_diff_reqdown) :
            signal += self.rsi_signalupdown
        if (RSI<self.rsi_reqdown) & (RSI_diff>self.rsi_diff_requp) :
            signal += self.rsi_signaldownup
        if (RSI<self.rsi_reqdown) & (RSI_diff<self.rsi_diff_reqdown) :
            signal += self.rsi_signaldowndown
                
        #signal requirements
        if signal>200.:
            proportion_to_hold = 1.
        elif signal<-200.:
            proportion_to_hold = 0.
        else :
            proportion_to_hold = last_recommendation
        return proportion_to_hold, timestamp, self.calculate_prices_variation(open_prices)
    
    def get_candles_needed_for_running(self):
        """ Returns the numbers of candles needed to process the strategy """
        return 100
    
#########################################################################

class TwoStrategiesCombination(Holding):
    def __init__(self, base_currency: str, quote_currency: str, time_frame: int,
                 base_strategy,
                 exp_strategy,
                 prices_repository=BinancePricesRepository(),
                 **kwargs):
        
        BaseStrategy = eval(base_strategy)
        ExpStrategy = eval(exp_strategy)
        
        self.base_strategy = BaseStrategy(base_currency, quote_currency, time_frame,prices_repository,**kwargs)
        self.exp_strategy = ExpStrategy(base_currency, quote_currency, time_frame,prices_repository,**kwargs)    
    
    def run(self,last_recommendation: float = 0, **kwargs):
        
        proportion_to_hold_base, timestamp, prices_variation = self.base_strategy.run(last_recommendation = last_recommendation)
        action_exp, timestamp, prices_variation = self.exp_strategy.get_action(last_recommendation)
        
        #Acciones combinadas
        hold_currency = (
                (proportion_to_hold_base == 1 and action_exp == 1) 
                or (proportion_to_hold_base == 1 and action_exp == 0)
                )
        proportion_to_hold = 1 if hold_currency else 0
     
        return proportion_to_hold, timestamp, prices_variation

#####################################################################

class ChandelierExitStrategy(Holding):
    
    MAX_NUMBER_OF_CANDLES = 100 #Cantidad normal de velas para calcular todos los indicadores
    
    def __init__(self, base_currency: str, quote_currency: str, time_frame: int,
                 prices_repository=BinancePricesRepository(),
                 **kwargs):
        super().__init__(base_currency, quote_currency, time_frame, prices_repository)
        self.mean = int(kwargs["mean"])
        self.chand_window = int(kwargs["chand_window"])
        self.mult_high = kwargs["mult_high"]
        self.mult_low = kwargs["mult_low"] 
    
    def calculate_true_range(self, df):
        df['tr1'] = df["high"] - df["low"]
        df['tr2'] = abs(df["high"] - df["close"].shift(1))
        df['tr3'] = abs(df["low"] - df["close"].shift(1))
        df['TR'] = df[['tr1','tr2','tr3']].max(axis=1)
        df.loc[df.index[0],'TR'] = 0
        return df

    def calculate_average_true_range(self, df):
        df = self.calculate_true_range(df)
        df['ATR'] = 0
        df.loc[df.index[self.mean],'ATR'] = round( df.loc[df.index[1:self.mean+1],"TR"].rolling(window=self.mean).mean()[-1], 4)
        const_atr = (self.mean-1)/self.mean
        const_tr = 1/self.mean
        ATR=df["ATR"].values
        TR=df["TR"].values
        for index in range(self.mean+1, len(df)):
            ATR[index]=ATR[index-1]*const_atr+TR[index]*const_tr
        df["ATR"]=ATR
        return df
    
    def calculate_chandelier_exits(self, df):
        df = self.calculate_average_true_range(df)
        df["chandelier_high"] = df['close']
        df["chandelier_low"] = df['close']
        df.loc[df.index[self.chand_window+1:],"chandelier_low"] = df.loc[df.index[self.chand_window+1:],"low"].rolling(window=self.chand_window).min() + self.mult_low * df["ATR"][self.chand_window+1:]
        df.loc[df.index[self.chand_window+1:],"chandelier_high"] = df.loc[df.index[self.chand_window+1:],"high"].rolling(window=self.chand_window).max() - self.mult_high * df["ATR"][self.chand_window+1:]
        return df

    def get_action(self, last_recommendation:float = 0):
        candles_data = self.get_candles_data()
        df = candles_data[['open','high','low','close']]
        timestamp = df.index[-1]    
        df = self.calculate_chandelier_exits(df)
        df["action"] = 0
        df["action"] = np.where(df["close"] > df["chandelier_low"], 1, df["action"])
        df["action"] = np.where(df["close"] < df["chandelier_high"], -1, df["action"])
        df["action"] = df["action"].shift(1) 
        return df["action"].iloc[-1], timestamp, self.calculate_prices_variation(df['open'])
    
    def run(self,last_recommendation: float = 0, **kwargs):
        action, timestamp, prices_variation = self.get_action(last_recommendation)
        if action == 1:
            proportion_to_hold = 1 
        else:
            proportion_to_hold = 0
        return proportion_to_hold, timestamp, prices_variation
         
    def get_candles_needed_for_running(self):
        """ Returns the numbers of candles needed to process the strategy """
        return self.MAX_NUMBER_OF_CANDLES
    
    

#####################################################################

class AccumulatedVolumeStrategy(Holding):
    
    def __init__(self, base_currency: str, 
                 quote_currency: str, 
                 time_frame: int,
                 prices_repository=BinancePricesRepository(),
                 **kwargs):
        super().__init__(base_currency, quote_currency, time_frame, prices_repository)
        self.obs_accum = int(kwargs["obs_accum"])
        self.target_window = int(kwargs["target_window"])
        self.volume_mean = int(kwargs["volume_mean"])
        self.volume_mean_shift = kwargs["volume_mean_shift"]
        self.bb_periods = int(kwargs["bb_periods"])
        self.std_bb = kwargs["std_bb"]
    
    def set_market_direction(self, open_prices):
        bb_high = bollinger_hband(open_prices, window=self.bb_periods, window_dev=self.std_bb, fillna=False).values
        bb_low = bollinger_lband(open_prices, window=self.bb_periods, window_dev=self.std_bb, fillna=False).values
        direction = np.zeros(bb_low.size)
        direction = np.where(open_prices > bb_high, 1, direction)
        direction = np.where(open_prices < bb_low, -1, direction)
        return direction
    
    def get_action(self,last_recommendation:float = 0):
        candles_data = self.get_candles_data()
        open_prices = candles_data.open
        volume = candles_data.shift(1).volume
        timestamp = open_prices.index[-1]
        
        volume_acum = volume.rolling(self.obs_accum).sum().fillna(0)
        volume_mean = volume_acum.ewm(span=self.volume_mean, adjust=False).mean().fillna(0) * self.volume_mean_shift       
        direction = self.set_market_direction(open_prices)
        candles_data["action"] = 0
        candles_data['action'] = np.where((volume_acum > volume_mean), direction, 0)
        candles_data = candles_data.reset_index()
        signals = candles_data.loc[(candles_data['action'] != candles_data['action'].shift(1)) & (candles_data['action'] != 0)]        

        for index in signals.index:
            candles_data.loc[index:index+self.target_window,'action'] = signals.loc[index,'action']
        
        return candles_data["action"].iloc[-1], timestamp,self.calculate_prices_variation(open_prices)
    
    def run(self,last_recommendation: float = 0, **kwargs):
        action,timestamp,prices_variation = self.get_action(last_recommendation)
        if action == 1:
            proportion_to_hold = 1 
        else:
            proportion_to_hold = 0
        return proportion_to_hold, timestamp, prices_variation
         
    def get_candles_needed_for_running(self):
        """ Returns the numbers of candles needed to process the strategy """
        return 40

#################################################################
        
class BBvolumeStrategy(Holding):
    
    def __init__(self, base_currency: str, 
                 quote_currency: str, 
                 time_frame: int,
                 prices_repository=BinancePricesRepository(),
                 **kwargs):
        super().__init__(base_currency, quote_currency, time_frame, prices_repository)
        self.time_frame = time_frame
        self.bb_periods = int(kwargs['bb_periods'])
        self.bb_mult_factor = kwargs['bb_mult_factor']
        self.vol_periods = int(kwargs['vol_periods']) 
        self.vol_mult_factor_down = kwargs['vol_mult_factor_down'] 
        self.vol_mult_factor_up = kwargs['vol_mult_factor_up']
    
    def get_action(self,candles_data,last_recommendation:float = 0):
        candles_data["volume"]=candles_data["volume"].shift(1)
        candles_data["high"]=candles_data["high"].shift(1)
        candles_data["low"]=candles_data["low"].shift(1)
        candles_data["sma"] = candles_data["open"].rolling(self.bb_periods).mean()
        candles_data["std"] = candles_data["open"].rolling(self.bb_periods).std()
        candles_data["bbu"] = candles_data["sma"] + \
            self.bb_mult_factor*candles_data["std"]
        candles_data["bbl"] = candles_data["sma"] - \
            self.bb_mult_factor*candles_data["std"]
        candles_data["kcu"] = keltner_channel_hband(\
                    high=candles_data["high"],\
                    low=candles_data["low"],\
                    close=candles_data["open"],window=self.bb_periods)
        candles_data["kcl"] = keltner_channel_lband(\
                    high=candles_data["high"],\
                    low=candles_data["low"],\
                    close=candles_data["open"],window=self.bb_periods)
        candles_data["vol_change"] = candles_data["volume"].rolling(self.vol_periods).mean()
        
        candles_data["action"] = np.nan
        candles_data["action"] = np.where((candles_data["bbu"]<candles_data["close"]) & \
             (candles_data["kcu"]<candles_data["close"]) & \
             ((candles_data["bbu"]>candles_data["close"].shift(1)) |\
             (candles_data["kcu"]>candles_data["close"].shift(1))) &\
             (candles_data["volume"]>self.vol_mult_factor_up*candles_data["vol_change"])\
             , 2,candles_data["action"])
        candles_data["action"] = np.where((candles_data["bbl"]>candles_data["close"]) & \
             (candles_data["kcl"]>candles_data["close"]) & \
             ((candles_data["bbl"]<candles_data["close"].shift(1)) |\
             (candles_data["kcl"]<candles_data["close"].shift(1))) &\
             (candles_data["volume"]<self.vol_mult_factor_down*candles_data["vol_change"])\
             ,1, candles_data["action"])
        candles_data["action"] = np.where((candles_data["bbu"]<candles_data["close"]) &\
             (candles_data["kcu"]<candles_data["close"]) &\
             ((candles_data["bbu"]>candles_data["close"].shift(1)) |\
             (candles_data["kcu"]>candles_data["close"].shift(1))) &\
             (candles_data["volume"]<self.vol_mult_factor_up*candles_data["vol_change"])\
             ,-1, candles_data["action"])
        candles_data["action"] = np.where((candles_data["bbl"]>candles_data["close"]) &\
             (candles_data["kcl"]>candles_data["close"]) &\
             ((candles_data["bbl"]<candles_data["close"].shift(1)) |\
             (candles_data["kcl"]<candles_data["close"].shift(1))) &\
             (candles_data["volume"]>self.vol_mult_factor_down*candles_data["vol_change"])\
             , -2, candles_data["action"])
       
        status = candles_data["action"].ffill()
        candles_data["action"] = np.where((status==2) &\
             (candles_data["sma"]>=candles_data["close"])
             ,0, candles_data["action"])
        candles_data["action"] = np.where((status==-2) &\
             (candles_data["sma"]<=candles_data["close"])
             ,0, candles_data["action"])
         
        candles_data["action"]=candles_data["action"].replace(2,1).replace(-2,-1)
        candles_data["action"]=candles_data["action"].shift(1).ffill().replace(np.nan,0)
        return candles_data["action"].iloc[-1]
    
    def run(self,last_recommendation: float = 0, **kwargs):
        candles_data = self.get_candles_data()
        timestamp = candles_data.index[-1]
        action = self.get_action(candles_data,last_recommendation)
        if action == 1:
            proportion_to_hold = 1 
        elif action == -1:
            proportion_to_hold = 0
        else :
            proportion_to_hold = last_recommendation
        return proportion_to_hold, timestamp, self.calculate_prices_variation(candles_data["open"].values)
         
    def get_candles_needed_for_running(self):
        """ Returns the numbers of candles needed to process the strategy """
        return (2*60*24*7)//self.time_frame

#################################################################

class LongTraillingStopLoss(Holding):
    
    MAX_NUMBER_OF_CANDLES = 100
    OBS_FOR_EXPIRATION_OF_PIVOT_ZONES = 168 #One week of hourly observations
    def __init__(self, base_currency: str, 
                 quote_currency: str, 
                 time_frame: int,
                 prices_repository=BinancePricesRepository(),
                 **kwargs):
        super().__init__(base_currency, quote_currency, time_frame, prices_repository)
        self.time_frame = time_frame
        self.obs_pps = int(kwargs['obs_pps'])
        self.mean_atr = int(kwargs['mean_atr'])
        self.umbral = 0.05
        
    def isSupport(self, df, i, n):
        if (((df.close[i] < df.close[i-n:i]).unique()).any() == True) or (((df.close[i] < df.close[i+1:i+1+n]).unique()).any() == True):
            return False
        else:
            return True if i>n else False
        
    def isResistance(self, df, i, n):
        if (((df.close[i] > df.close[i-n:i]).unique()).any() == True) or (((df.close[i] > df.close[i+1:i+1+n]).unique()).any() == True):
            return False
        else:
            return True if i>n else False
    
    def isFarFromLevel(self, l, s, levels):
        return np.sum([abs(l-x[2]) < s for x in levels]) == 0
    
    def isCloseFromResistance(self, df, levels):
        #1) que la diferencia entre el nivel importante y el close sea menor a X ATR
        #2) que el precio este por debajo del nivel, indicando ser resistencia long
        #3) que el nivel importante este generado con informacion del pasado y no del futuro
        count_signals = lambda row: np.sum(
                                    [(abs(x[2] - row['close']) <= row['ATR']/4 
                                        and row['close'] < x[2] 
                                        and row.name > x[0])
                                        for x in levels]
                                ) 
        
        return df.apply(lambda row: count_signals(row) != 0, axis = 1)
        
    def isCloseFromSupport(self, df, levels):
        #1) que la diferencia entre el nivel importante y el close sea menor a un ATR
        #2) que el precio este por encima del nivel, indicando ser soporte long
        #3) que el nivel importante este generado con informacion del pasado y no del futuro
        count_signals = lambda row: np.sum(
                                    [(abs(x[2] - row['close']) <= row['ATR']/4 
                                        and row['close'] > x[2] 
                                        and row.name > x[0])
                                        for x in levels]
                                ) 
        
        return df.apply(lambda row: count_signals(row) != 0, axis = 1)
    
    def combine_level(self, l, s, levels):
        obs, rep, prices = np.empty(0,dtype=int), np.empty(0,dtype=int), np.empty(0,dtype=int)
        for x in levels:
              if abs(l-x[2]) < s:
                  obs, rep, prices = np.append(obs, x[0]), np.append(rep, x[1]), np.append(prices, x[2])
        levels = [x for x in levels if x[0] not in obs] #Elimina los duplicados
        levels.append((obs.min(), rep.sum()+1, np.append(prices, l).mean()))
        return levels
        
    def find_important_levels(self, df):
        levels, pps = [], []
        for i in range(self.obs_pps, df.shape[0]-self.obs_pps):
            if self.isSupport(df,i, self.obs_pps):
                l, s = df['close'][i], df["ATR"][i]/2
                pps.append((df.index[i],l))
                if self.isFarFromLevel(l, s, levels):
                    levels.append((df.index[i],0,l))
                else:
                    levels = self.combine_level(l, s, levels)
            elif self.isResistance(df,i, self.obs_pps):
                l, s = df['close'][i], df["ATR"][i]/2
                pps.append((df.index[i],l))
                if self.isFarFromLevel(l, s, levels):
                    levels.append((df.index[i],0,l))
                else:
                    levels = self.combine_level(l, s, levels)
        levels = [i for i in levels if i[1] != 0 or i[0] > df.index[-self.OBS_FOR_EXPIRATION_OF_PIVOT_ZONES]]
        return levels, pps
                
    def find_important_areas(self, df):
        areas, pps = [], []
        for i in range(self.obs.pps, df.shape[0]-self.obs.pps):
            if self.isSupport(df,i, self.obs.pps):
                l, s = df['close'][i], df["ATR"][i]/2
                pps.append((df.index[i],l))
                if self.isFarFromArea(l, s, areas):
                    areas.append((df.index[i],0,l,l))
                else:
                    areas = self.combine_area(l, s, areas)
            elif self.isResistance(df,i, self.obs.pps):
                l, s = df['close'][i], df["ATR"][i]/2
                pps.append((df.index[i],l))
                if self.isFarFromArea(l, s, areas):
                    areas.append((df.index[i],0,l,l))
                else:
                    areas = self.combine_area(l, s, areas)
        areas = [i for i in areas if i[1] != 0 or i[0] > df.index[-self.OBS_FOR_EXPIRATION_OF_PIVOT_ZONES]]
        return areas, pps
    
    def find_pivot_points(self, df):
        pps = []
        for i in range(self.obs_pps, df.shape[0]-self.obs_pps):
            if self.isSupport(df,i, self.obs_pps):
                l = df['close'][i]
                pps.append((df.index[i],l))
            elif self.isResistance(df,i, self.obs_pps):
                l = df['close'][i]
                pps.append((df.index[i],l))
        return pps
    
    def find_important_areas_expiration(self, df):
        df = self.calculate_average_true_range(df)
        pps = self.find_pivot_points(df)
        df['areas'] = 0
        df['areas'] = df['areas'].astype(object)
        for i in range(len(df)):
            df.at[df.index[i], 'areas'] = self.combine_area_expiration(i, df, pps)
        return df, pps
            
    def combine_area_expiration(self, i, df, pps):
        idx_start, idx_end = df.index[0] if i<200 else df.index[i-200], df.index[i]
        pps_to_consider = [pp for pp in pps if pp[0] >= idx_start and pp[0] <= idx_end]
        
        areas, s = [], df[idx_start: idx_end]['ATR'].mean()
        for pp in pps_to_consider:
            if self.isFarFromLevel(pp[1], s, areas):
                areas.append((pp[0],0,pp[1],pp[1]))
            else:
                areas = self.combine_area(pp[1], s, areas)
        return areas
    
    def combine_area(self, l, s, areas):
        obs, rep, maxs, mins = np.empty(0,dtype=int), np.empty(0,dtype=int), np.empty(0,dtype=int), np.empty(0,dtype=int)
        for x in areas:
              if (abs(l-x[2]) < s or abs(l-x[3]) < s):
                  obs, rep, maxs, mins = np.append(obs, x[0]), np.append(rep, x[1]), np.append(maxs, x[2]), np.append(mins, x[3])
        areas = [x for x in areas if x[0] not in obs] #Elimina los duplicados
        areas.append((obs.min(), rep.sum()+1, np.append(maxs, l).max(), np.append(mins, l).min()))
        return areas
    
    def isFarFromArea(self, l, s, areas):
        return (np.sum([abs(l-x[2]) < s for x in areas]) == 0 or
                np.sum([abs(l-x[3]) < s for x in areas]) == 0)
    
    def isCloseFromLowerArea(self, df, areas):
        #1) que la diferencia entre el nivel importante y el close sea menor a X ATR
        #2) que el precio este por debajo del nivel, indicando ser resistencia long
        #3) que el nivel importante este generado con informacion del pasado y no del futuro
        count_signals = lambda row: np.sum(
                                    [(abs(x[3] - row['close']) <= row['ATR']/4 
                                        and row['close'] < x[3] 
                                        and row.name > x[0])
                                        for x in areas]
                                    ) 
        
        return df.apply(lambda row: count_signals(row) != 0, axis = 1)
        
    def isCloseFromUpperArea(self, df, areas):
        #1) que la diferencia entre el nivel importante y el close sea menor a un ATR
        #2) que el precio este por encima del nivel, indicando ser soporte long
        #3) que el nivel importante este generado con informacion del pasado y no del futuro
        count_signals = lambda row: np.sum(
                                    [(abs(x[2] - row['close']) <= row['ATR']/4 
                                        and row['close'] > x[2] 
                                        and row.name > x[0])
                                        for x in areas]
                                    ) 
        
        return df.apply(lambda row: count_signals(row) != 0, axis = 1)
        
    def calculate_true_range(self, df):
        df['tr1'] = df["high"] - df["low"]
        df['tr2'] = abs(df["high"] - df["close"].shift(1))
        df['tr3'] = abs(df["low"] - df["close"].shift(1))
        df['TR'] = df[['tr1','tr2','tr3']].max(axis=1)
        df.loc[df.index[0],'TR'] = 0
        return df
    
    def calculate_average_true_range(self, df):
        df = self.calculate_true_range(df)
        df['ATR'] = 0
        try:
            df.loc[df.index[self.mean_atr],'ATR'] = round( df.loc[df.index[1:self.mean_atr+1],"TR"].rolling(window=self.mean_atr).mean()[self.mean_atr], 4)
        except:
            df.loc[df.index[self.mean_atr],'ATR'] = round( df.loc[df.index[1:self.mean_atr+1],"TR"].rolling(window=self.mean_atr).mean()[-1], 4)
        const_atr = (self.mean_atr-1)/self.mean_atr
        const_tr = 1/self.mean_atr
        ATR=df["ATR"].values
        TR=df["TR"].values
        for index in range(self.mean_atr+1, len(df)):
            ATR[index]=ATR[index-1]*const_atr+TR[index]*const_tr
        df["ATR"]=ATR
        return df
    
    def long_trailling_stop(self, df):
        df['ret'] = df.close.pct_change()
        df['index'] = df.index
        table = df.loc[(df['action'] != df['action'].shift()) & (df['action'].notna())]
        df['ret_accum'] = 0
        for row in range(len(table)):
            if table.action.iloc[row] == 1:
                try:
                    index_start, index_end = table.index[row], table.index[row+1]
                except:
                    index_start, index_end = table.index[row], df.index[len(df)-1]
                df['ret_accum'][index_start:index_end] = df.ret[index_start:index_end].cumsum()  
                df.loc[df['ret_accum'] < -self.umbral, 'action'] = 0        
        return df
    
    def get_action(self, last_recommendation: float = 0):
        df = self.get_candles_data()
        timestamp = df.open.index[-1]
    
        df, _ = self.find_important_areas_expiration(df) 
        '''
        Se vende si:
            1) No se está dentro de una zona pivote, [condicion de lateralidad]
            2) Se tiene al menos 1 ATRs de recorrido al alza hasta la zona pivote superior más cercana. [condicion de recorrido alcista]
        '''
        df['action'] = 0
        df['condition1'],df['condition2'] = 0,1
        for row in range(1,len(df)):
            df.condition1[row] = 1 not in [1 for area in df.areas[row] if (area[2] >= df.close[row] >= area[3] and df.index[row]>area[0])]                          
            try:
                df.condition2[row] = (( 1/2*df.ATR[row] < np.min([df.close[row]-area[2] for area in df.areas[row] if (df.close[row]-area[2]>0 and area[1]>0)]) ) )
            except:
                df.condition2[row] = True
        df['action'] = np.where((df['condition1']==1) &
                                (df['condition2']==1),
                                1, df['action'])
        df = self.long_trailling_stop(df)
        
        if (df["action"]==0).all():
            return df["action"]
        else:
            df['action'] = df['action'].replace(0,np.nan)
            df['action'] = df['action'].fillna(method='ffill').shift(1)
        return df["action"].iloc[-1], timestamp, self.calculate_prices_variation(df.open)
    
    def run(self, last_recommendation: float = 0, **kwargs):
        action, timestamp, prices_variation = self.get_action(last_recommendation)
        if action == 1:
            proportion_to_hold = 1 
        else:
            proportion_to_hold = 0
        return proportion_to_hold, timestamp, prices_variation
    
    def get_candles_needed_for_running(self):
        """ Returns the numbers of candles needed to process the strategy """
        return self.MAX_NUMBER_OF_CANDLES


class VolumeProfileStrategy(Holding):

    max_number_of_candles = 1000
    
    def __init__(self, 
        base_currency: str, 
        quote_currency: str,
        time_frame: int, 
        bb_periods: int, 
        bb_std_dev: float, 
        n_candles_volume: int,  
        cte_vol_1: float,
        cte_vol_2 : float,
        threshold_bb_rate: float,
        prices_repository=BinancePricesRepository()):

        super().__init__(base_currency, quote_currency, time_frame, prices_repository)         
        self.bb_periods = int(bb_periods)
        self.bb_std_dev = bb_std_dev
        self.n_candles_volume = int(n_candles_volume)        
        self.cte_vol_1 = cte_vol_1
        self.cte_vol_2 = cte_vol_2
        self.threshold_bb_rate = threshold_bb_rate
        
    def run(self, last_recommendation: float =0, **kwargs):
        """ 
        return strategy status. Value 1 means hold, 0 means do not hold. 
        """

        candles_data = self.get_candles_data()
        candles_data = self.increase_values_to_significant_magnitudes(candles_data, self.quote_currency)

        open_prices = candles_data.open
        volume = candles_data.volume

        open_prices_pct = open_prices.pct_change()

        current_variables = self.get_dictionary_of_value_of_current_variables(open_prices, volume.shift(1))

        current_bb_rate = current_variables['bb_rate']
        current_macd_diff = current_variables['macd_diff'] 
        current_volume_sma = current_variables['volume_sma']
            
        timestamp = open_prices.index[-1]
        timestamp_shift_1 = open_prices.index[-2]
        timestamp_shift_2 = open_prices.index[-3]
    
        #Volume Profile!
        value_area_low, value_area_high = self.get_market_profile_for_a_daily_slice(timestamp)

        if timestamp_shift_1.day == timestamp.day:
            value_area_high_shift_1 = value_area_high
            if timestamp_shift_2.day == timestamp_shift_1.day:
                value_area_high_shift_2 = value_area_high_shift_1
            else:
                _, value_area_high_shift_2 = self.get_market_profile_for_a_daily_slice(timestamp_shift_2)
        else:
            _, value_area_high_shift_1 = self.get_market_profile_for_a_daily_slice(timestamp_shift_1)
            value_area_high_shift_2 = value_area_high_shift_1

        #Desicion criteria  
        #Buy
        hold1 = (open_prices[-1] > value_area_high)\
            and (open_prices[-2] > value_area_high_shift_1)\
            and (open_prices[-3] > value_area_high_shift_2)\

        hold2 = (open_prices_pct[-1] > 0)\
            and  (volume[-1] > self.cte_vol_1 * current_volume_sma)\
            and  (open_prices[-1] > value_area_low)\
            and  (current_macd_diff > 0) \
            and  (open_prices_pct[-2] > 0)\

        hold3 = (open_prices[-1] > value_area_high)


        #Sell    
        not_hold = (open_prices[-1] < value_area_low)\
                and  (open_prices_pct[-1] > -0.05)\
                and  (open_prices_pct[-1] < 0)\
                and  (volume[-1] > self.cte_vol_2 * current_volume_sma)

        hold = ((hold2 or hold3) and (current_bb_rate <= self.threshold_bb_rate))\
            or (hold1 and (current_bb_rate > self.threshold_bb_rate))
        
        if hold:
            recommendation = 1
        elif not_hold:
            recommendation = 0
        else:
            recommendation = last_recommendation
        
        return recommendation, timestamp, self.calculate_prices_variation(open_prices)

    def get_dictionary_of_value_of_current_variables(self, open_prices, volume):
        
        variable = dict()
        #Bollinger Bands
        bb_lband = bollinger_lband(open_prices, window=self.bb_periods, window_dev=self.bb_std_dev)[-1]
        bb_hband = bollinger_hband(open_prices, window=self.bb_periods, window_dev=self.bb_std_dev)[-1]
        #Rate Bollinger Bands
        variable['bb_rate'] = (bb_hband - bb_lband)/bb_lband
        #MACD        
        variable['macd_diff'] = macd_diff(open_prices, window_slow=24, window_fast=12, window_sign=9, fillna = False)[-1]
        #Volume
        variable['volume_sma'] = volume.rolling(window=self.n_candles_volume).mean()[-1]
        
        return variable
    
    def get_market_profile_for_a_daily_slice(self, datetime):
        
        #Candles 5 min in order to calculate volume profile!! 
        time_frame_for_volume_profile = 5
        n_candles = self.get_candles_needed_for_running()
        candles_for_volume_profile = self.prices_repository.get_candles(
                        self.base_currency, 
                        self.quote_currency, 
                        MINUTES_TO_BINANCE_INTERVAL[time_frame_for_volume_profile], 
                        n_candles)
        candles_for_volume_profile = self.increase_values_to_significant_magnitudes(candles_for_volume_profile, self.quote_currency)

        df = pd.DataFrame()
        df['Close'] = candles_for_volume_profile.close
        df['Open'] = candles_for_volume_profile.open
        df['High'] = candles_for_volume_profile.high
        df['Low'] = candles_for_volume_profile.low
        df['Volume'] = candles_for_volume_profile.volume

        datetime_yesterday = datetime - pd.Timedelta(1, 'd')
        datetime_yesterday = datetime_yesterday.strftime("%Y-%m-%d")

        df = df.loc[datetime_yesterday]

        market_profile_ = MarketProfile(df)
        mp_slice = market_profile_[df.index[0] : df.index[-1]]

        return mp_slice.value_area[0], mp_slice.value_area[1]
    
    def increase_values_to_significant_magnitudes(self, candles_data, quote):
        """
        If quote_currency is BTC, the prices are too small, so we need to multiplier for 1e3 in order to avoid 
        numerical errors. 
        """
        if quote == 'BTC':
            numeric_columns = candles_data._get_numeric_data().columns
            candles_data[numeric_columns] = 10000*candles_data[numeric_columns]

        return candles_data 
    
    def get_candles_needed_for_running(self):
        return self.max_number_of_candles 

