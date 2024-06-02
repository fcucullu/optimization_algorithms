import pandas as pd
import numpy as np
from datetime import datetime

from utils.candlestick import BinancePricesRepository
from utils.pair_strategies import Holding
from utils.tendency_checker import TendencyChecker, BULL_MARKET, BEAR_MARKET, LATERAL_MARKET
TC = TendencyChecker()


class BullMarketNowBetterThanBefore(Holding):

    LONG = 1
    CLOSE_LONG = 0

    CANDLES_NEEDED_FOR_RUNNING = {
        60: 24 * 10, # Two days + Some extra for dealing with rollings. 
        15: (24 * 10) * 4
    }

    N_FOR_RECENT_LOWS = {
        60: 6  ,
        15: 6 * 4
    }

    N_FOR_PAST_LOWS = {
        60: 12,
        15: 12 * 4,
    }

    WINDOWS_A = {
        60: 36,
        15: 36 * 4,
    }

    WINDOWS_B = {
        60: 48,
        15: 48 * 4,
    }

    def __init__(self, 
        base_currency: str, 
        quote_currency: str,
        time_frame: int, 
        bull_market_factor: float,
        bear_market_factor: float,
        active_operation_threshold: float = 0,
        threshold_factor: float = 1.027,
        prices_repository=BinancePricesRepository()):

        super().__init__(base_currency, quote_currency, time_frame, prices_repository)         

        self.bull_market_factor = bull_market_factor
        self.bear_market_factor = bear_market_factor 
        self.active_operation_threshold = active_operation_threshold
        self.threshold_factor = threshold_factor

    def run(self, last_recommendation:float = 0):
        candles = self.get_candles_data()
        timestamp = candles.index[-1]
        price_variation = self.calculate_prices_variation(candles['open'])

        recommendation = self.calculate_recommendation(candles)
        if recommendation is None:
            recommendation = last_recommendation

        return recommendation, timestamp, price_variation

    def calculate_recommendation(self, original_candles):
        candles = self.process_candles(original_candles[['open', 'low', 'high']])
        candles = self.check_threshold_and_takeprofit(candles) 

        current_state = candles['signal'].iloc[-1]
        previous_state = candles['signal'].iloc[-2]
        change_of_state =  current_state != previous_state
        recommendation = None
        if change_of_state:
            if candles['signal'].iloc[-1] == self.LONG:
                recommendation = 1
            elif candles['signal'].iloc[-1] == self.CLOSE_LONG:
                recommendation = 0
            else:
                raise ValueError("Signal is not defined")
        
        return recommendation

    def process_candles(self, original_candles):
        candles = original_candles.copy()
        # el shift decarta el mínimo actual y lo reemplaza por el anterior

        n_for_recents_low = self.N_FOR_RECENT_LOWS[self.time_frame]
        n_for_past_lows = self.N_FOR_PAST_LOWS[self.time_frame]
        candles['recent_lows'] = candles.low.rolling(window=n_for_recents_low).mean().shift() 
        candles['past_lows'] = candles.low.shift(n_for_recents_low).rolling(window=n_for_past_lows).mean().shift()
        
        # past_lows modification
        window_a = self.WINDOWS_A[self.time_frame]
        window_b = self.WINDOWS_B[self.time_frame]        
        candles['returns'] = candles.open.pct_change()
        candles['last_return_A'] = (candles['returns']+1).rolling(window=window_a).agg(lambda x : x.prod())
        candles['last_return_B'] = (candles['returns']+1).rolling(window=window_b).agg(lambda x : x.prod())
        candles.dropna(inplace=True)

        candles['factor_to_reduce_past_lows'] = np.where(
            (candles['last_return_A'] > 1.01) \
            & (candles['last_return_B'] > 1.01),
            self.bull_market_factor,
            self.bear_market_factor,
        )
        
        # acá usamos el tamaño de la observaciones de recent lows para suavizar  
        candles['factor_to_reduce_past_lows'] = candles['factor_to_reduce_past_lows'].rolling(n_for_recents_low).mean()
        candles['past_lows'] = candles['past_lows'] * candles['factor_to_reduce_past_lows']

        now_is_better_than_before = (candles.recent_lows > candles.past_lows)
        long_condition = now_is_better_than_before

        candles['basic_signal'] = np.where(
            long_condition,
            self.LONG, 
            self.CLOSE_LONG, 
        ) 

        return candles

    def check_threshold_and_takeprofit(self, candles):
        candles['signal'] = candles['basic_signal']

        if candles.basic_signal.iloc[-1] != self.LONG:
            return candles
        
        # estamos adentro, tenemos que filtrar la data del último trade
        trade_start_offset = self.get_trade_start_offset(candles)

        # Inicializamos valores
        ACTIVE_TP_FACTOR = 1.025
        STOP_LOSS_FACTOR = 0.97
        RESET_TRESHOLD_FACTOR = 1.025

        candles['threshold_by_mins_values'] = candles.low.rolling(24*2).mean() * self.threshold_factor
        threshold_by_mins_values = candles['threshold_by_mins_values'].iloc[trade_start_offset] 
        active_operation_threshold = max(threshold_by_mins_values, self.active_operation_threshold)

        if candles.open.iloc[trade_start_offset] > active_operation_threshold:
            buy_price = candles.open.iloc[trade_start_offset]
            stop_loss = candles.open.iloc[trade_start_offset] * STOP_LOSS_FACTOR
            active_tp = candles.open.iloc[trade_start_offset] * ACTIVE_TP_FACTOR        
            local_threshold = 0
            max_price = candles.high.iloc[trade_start_offset]
            gain_factor = 0
            already_in = True
        else:
            buy_price = None
            local_threshold = active_operation_threshold
            stop_loss = None
            active_tp = None
            max_price = None
            gain_factor = None
            already_in = False            


        candles['buy_price'] = None
        candles['local_threshold'] = None
        candles['max_price'] = None
        candles['gain_factor'] = None
        candles['stop_loss'] = None
        candles['active_tp'] = None        

        # valor auxiliar para setear el DF
        signal_col_number = candles.columns.get_loc("signal")
        buy_price_col_number = candles.columns.get_loc('buy_price')
        local_threshold_col_number = candles.columns.get_loc('local_threshold')
        max_price_col_number = candles.columns.get_loc('max_price')
        gain_factor_col_number = candles.columns.get_loc('gain_factor')
        stop_loss_col_number = candles.columns.get_loc('stop_loss')
        active_tp_col_number = candles.columns.get_loc('active_tp')


        for i in range(trade_start_offset, len(candles)):

            if buy_price is not None: # Estoy holdeando

                # Default value.
                candles.iat[i,signal_col_number] = self.LONG

                # Chequeo si salgo. Sí salgo, inicializo todo para operar dsp                

                # el gain factor se usa para calcular el TP dinámico
                # dependiendo el tiempo puede ser hasta un 70% de la 
                # ganancia máxima ("maximo luego de entrar" - "precio de entrada")
                if candles.open.iloc[i] < stop_loss:
                    candles.iat[i,signal_col_number] = self.CLOSE_LONG
                    local_threshold = stop_loss * RESET_TRESHOLD_FACTOR
                    buy_price = None

                # Si estoy arriba del `active_tp`, actualizo el local_threshold
                if candles.open.iloc[i] > active_tp:   # <- (A)
                    gain_factor = min(gain_factor + 0.085, 0.7)  
                    max_price = max(candles.high.iloc[i-1], max_price)
                    local_threshold = buy_price + (max_price - buy_price) * gain_factor
                
                # Siempre tengo que estar arriba del `local_threshold`
                # Notar que se inicializa a 0 y esto solo cambia luego de 
                # una ocurrencia de (A)
                if candles.open.iloc[i] < local_threshold:
                    #se cierra la posición
                    candles.iat[i,signal_col_number] = self.CLOSE_LONG
                    buy_price = None
                    # subo en threhold por si tengo que volver a entrar
                    local_threshold = local_threshold * RESET_TRESHOLD_FACTOR  #
            else: 
                # dejé de holdear
                # default_value
                candles.iat[i,signal_col_number] = self.CLOSE_LONG

                # update del local_threshold en base al  threshold_by_mins_values
                if already_in:
                    local_threshold = max(local_threshold, candles['threshold_by_mins_values'].iloc[i])
                else:
                    local_threshold = candles['threshold_by_mins_values'].iloc[i]

                #Chequeo si tengo que entrar y si sí seteo los valores para el TP:
                if candles.open.iloc[i] > local_threshold:
                    already_in = True
                    candles.iat[i,signal_col_number] = self.LONG    
                    buy_price = candles.open.iloc[i]
                    max_price = buy_price
                    gain_factor = 0
                    stop_loss = buy_price * STOP_LOSS_FACTOR
                    active_tp = buy_price * ACTIVE_TP_FACTOR
                    local_threshold = 0

            candles.iat[i, buy_price_col_number] = buy_price 
            candles.iat[i, local_threshold_col_number] = local_threshold 
            candles.iat[i, max_price_col_number] = max_price 
            candles.iat[i, gain_factor_col_number] = gain_factor 
            candles.iat[i, stop_loss_col_number] =  stop_loss
            candles.iat[i, active_tp_col_number] =  active_tp
            

        return candles


    def get_trade_start_offset(self, candles):
        position = -1 # last element
        while candles['basic_signal'].iloc[position] == self.LONG:
            position -= 1

        return (len(candles) + position + 1) # position is negative

    def get_candles_needed_for_running(self):
        """ Returns the numbers of candles needed to process the strategy """
        return self.CANDLES_NEEDED_FOR_RUNNING[self.time_frame]



class BullMarketNowBetterThanBeforeDynamicParameters(BullMarketNowBetterThanBefore):
    
    def __init__(self, 
        base_currency: str, 
        quote_currency: str,
        time_frame: int, 
        active_operation_threshold: float = 0,
        prices_repository=BinancePricesRepository()):

        bull_market_factor, bear_market_factor, threshold_factor, self.side_market = 0,0,0,0

        super().__init__(base_currency, 
                         quote_currency, 
                         time_frame, 
                         bull_market_factor, 
                         bear_market_factor,
                         active_operation_threshold,
                         threshold_factor,
                         prices_repository)  

        self.base_currency = base_currency
        self.quote_currency = quote_currency
        self.time_frame = time_frame
        
    def process_candles(self, original_candles):
        candles = original_candles.copy()
        # el shift decarta el mínimo actual y lo reemplaza por el anterior

        n_for_recents_low = self.N_FOR_RECENT_LOWS[self.time_frame] 
        n_for_past_lows = self.N_FOR_PAST_LOWS[self.time_frame]
        candles['recent_lows'] = candles.low.rolling(window=n_for_recents_low).mean().shift() 
        candles['past_lows'] = candles.low.shift(n_for_recents_low).rolling(window=n_for_past_lows).mean().shift()
        
        # past_lows modification
        window_a = self.WINDOWS_A[self.time_frame]
        window_b = self.WINDOWS_B[self.time_frame]        
        candles['returns'] = candles.open.pct_change()
        candles['last_return_A'] = (candles['returns']+1).rolling(window=window_a).agg(lambda x : x.prod())
        candles['last_return_B'] = (candles['returns']+1).rolling(window=window_b).agg(lambda x : x.prod())
        candles.dropna(inplace=True)

        candles['bull_market_factor'] = 0.
        candles['bear_market_factor'] = 0.
        candles['threshold_factor'] = 0.
        candles['side_market'] = ''
        for i in range(60,len(candles)):
            side_market = TC.check_tendency_on_specific_date(timeserie=candles.close,
                                               date=candles.index[i],
                                               interval_width_in_days=1,
                                               minutes_in_obs=self.time_frame
                                               )
            if side_market == BULL_MARKET:
                candles['bull_market_factor'][i], candles['bear_market_factor'][i], candles['threshold_factor'] = .97, 1, 1.027
            elif (side_market == LATERAL_MARKET) or (side_market == BEAR_MARKET):
                candles['bull_market_factor'][i], candles['bear_market_factor'][i], candles['threshold_factor'] =  1, 1, 1.05
            candles['side_market'][i] = side_market
            
        candles['factor_to_reduce_past_lows'] = np.where(
            (candles['last_return_A'] > 1.01) \
            & (candles['last_return_B'] > 1.01),
            candles['bull_market_factor'],
            candles['bear_market_factor'],
        )
        
        # acá usamos el tamaño de la observaciones de recent lows para suavizar  
        candles['factor_to_reduce_past_lows'] = candles['factor_to_reduce_past_lows'].rolling(n_for_recents_low).mean()
        candles['past_lows'] = candles['past_lows'] * candles['factor_to_reduce_past_lows']

        now_is_better_than_before = (candles.recent_lows > candles.past_lows)
        long_condition = now_is_better_than_before

        candles['basic_signal'] = np.where(
            long_condition,
            self.LONG, 
            self.CLOSE_LONG, 
        ) 

        return candles

    def check_threshold_and_takeprofit(self, candles):
        candles['signal'] = candles['basic_signal']

        if candles.basic_signal.iloc[-1] != self.LONG:
            return candles
        
        # estamos adentro, tenemos que filtrar la data del último trade
        trade_start_offset = self.get_trade_start_offset(candles)

        # Inicializamos valores
        ACTIVE_TP_FACTOR = 1.025
        STOP_LOSS_FACTOR = 0.97
        RESET_TRESHOLD_FACTOR = 1.025

        candles['threshold_by_mins_values'] = candles.low.rolling(24*2).mean() * candles['threshold_factor']
        threshold_by_mins_values = candles['threshold_by_mins_values'].iloc[trade_start_offset] 
        active_operation_threshold = max(threshold_by_mins_values, self.active_operation_threshold)

        if candles.open.iloc[trade_start_offset] > active_operation_threshold:
            buy_price = candles.open.iloc[trade_start_offset]
            stop_loss = candles.open.iloc[trade_start_offset] * STOP_LOSS_FACTOR
            active_tp = candles.open.iloc[trade_start_offset] * ACTIVE_TP_FACTOR        
            local_threshold = 0
            max_price = candles.high.iloc[trade_start_offset]
            gain_factor = 0
            already_in = True
        else:
            buy_price = None
            local_threshold = active_operation_threshold
            stop_loss = None
            active_tp = None
            max_price = None
            gain_factor = None
            already_in = False            


        candles['buy_price'] = None
        candles['local_threshold'] = None
        candles['max_price'] = None
        candles['gain_factor'] = None
        candles['stop_loss'] = None
        candles['active_tp'] = None        

        # valor auxiliar para setear el DF
        signal_col_number = candles.columns.get_loc("signal")
        buy_price_col_number = candles.columns.get_loc('buy_price')
        local_threshold_col_number = candles.columns.get_loc('local_threshold')
        max_price_col_number = candles.columns.get_loc('max_price')
        gain_factor_col_number = candles.columns.get_loc('gain_factor')
        stop_loss_col_number = candles.columns.get_loc('stop_loss')
        active_tp_col_number = candles.columns.get_loc('active_tp')


        for i in range(trade_start_offset, len(candles)):

            if buy_price is not None: # Estoy holdeando

                # Default value.
                candles.iat[i,signal_col_number] = self.LONG

                # Chequeo si salgo. Sí salgo, inicializo todo para operar dsp                

                # el gain factor se usa para calcular el TP dinámico
                # dependiendo el tiempo puede ser hasta un 70% de la 
                # ganancia máxima ("maximo luego de entrar" - "precio de entrada")
                if candles.open.iloc[i] < stop_loss:
                    candles.iat[i,signal_col_number] = self.CLOSE_LONG
                    local_threshold = stop_loss * RESET_TRESHOLD_FACTOR
                    buy_price = None

                # Si estoy arriba del `active_tp`, actualizo el local_threshold
                if candles.open.iloc[i] > active_tp:   # <- (A)
                    gain_factor = min(gain_factor + 0.085, 0.7)  
                    max_price = max(candles.high.iloc[i-1], max_price)
                    local_threshold = buy_price + (max_price - buy_price) * gain_factor
                
                # Siempre tengo que estar arriba del `local_threshold`
                # Notar que se inicializa a 0 y esto solo cambia luego de 
                # una ocurrencia de (A)
                if candles.open.iloc[i] < local_threshold:
                    #se cierra la posición
                    candles.iat[i,signal_col_number] = self.CLOSE_LONG
                    buy_price = None
                    # subo en threhold por si tengo que volver a entrar
                    local_threshold = local_threshold * RESET_TRESHOLD_FACTOR  #
            else: 
                # dejé de holdear
                # default_value
                candles.iat[i,signal_col_number] = self.CLOSE_LONG

                # update del local_threshold en base al  threshold_by_mins_values
                if already_in:
                    local_threshold = max(local_threshold, candles['threshold_by_mins_values'].iloc[i])
                else:
                    local_threshold = candles['threshold_by_mins_values'].iloc[i]

                #Chequeo si tengo que entrar y si sí seteo los valores para el TP:
                if candles.open.iloc[i] > local_threshold:
                    already_in = True
                    candles.iat[i,signal_col_number] = self.LONG    
                    buy_price = candles.open.iloc[i]
                    max_price = buy_price
                    gain_factor = 0
                    stop_loss = buy_price * STOP_LOSS_FACTOR
                    active_tp = buy_price * ACTIVE_TP_FACTOR
                    local_threshold = 0

            candles.iat[i, buy_price_col_number] = buy_price 
            candles.iat[i, local_threshold_col_number] = local_threshold 
            candles.iat[i, max_price_col_number] = max_price 
            candles.iat[i, gain_factor_col_number] = gain_factor 
            candles.iat[i, stop_loss_col_number] =  stop_loss
            candles.iat[i, active_tp_col_number] =  active_tp
            

        return candles

