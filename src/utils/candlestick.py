import pandas as pd
import requests
from datetime import datetime, timedelta
from influxdb import DataFrameClient


class CandlestickRepository:
    TABLE_WITH_1MINUTE_CANDLES = 'candles'
    PREPROD_DB_NAME = 'xcapit'
    PREPROD_HOST = '10.10.17.166'
    PREPROD_PORT = 8086
    
    def __init__(self, db_name, host, port):
        self.db_name = db_name
        self.influx_client = DataFrameClient(host, port)
    
    @classmethod
    def default_repository(cls):
        return cls(cls.PREPROD_DB_NAME, cls.PREPROD_HOST, cls.PREPROD_PORT)
        
    def get_one_minute_candlestick(
        self, 
        pair: str, 
        exchange: str, 
        start_time: datetime, 
        end_time: datetime
    ) -> pd.DataFrame:
    
        """Function gets one minute candles data for `pair` from influxdb. 
        
        Arguments:
            pair {str} -- the pair of the data to request
            exchange {str} -- exchange used as source
            start_time {datetime} -- when the time interval starts
            end_time {datetime} -- when the time interval ends
        
        Returns:
            pd.DataFrame -- DataFrame with candles data.
        """
    
        query = f"""select * from {table} where 
        exchange=$exchange and pair=$pair 
        and $start_time <= time and time<$end_time"""

        bind_params = {
            'exchange': exchange,
            'pair': pair,
            'start_time': f"{start_time}".split('+')[0],
            'end_time': f"{end_time}".split('+')[0]
        }
        result = self.influx_client.query(
            query,
            bind_params=bind_params,
            database=self.db_name
        )
        return result.get(self.TABLE_WITH_1MINUTE_CANDLES, pd.DataFrame())
    
    def get_candlestick(
        self, 
        pair: str, 
        exchange: str, 
        size: int,
        start_time: datetime, 
        end_time: datetime,
    ) -> pd.DataFrame:
        """Function gets candles data for `pair` from influxdb with candles of
        size `size` minutes. 
        
        Arguments:
            pair {str} -- the pair of the data to request
            exchange {str} -- exchange used as source
            size {int} -- minutes used to construct the candles 
            start_time {datetime} -- when the time interval starts
            end_time {datetime} -- when the time interval ends
        
        Returns:
            pd.DataFrame -- DataFrame with candles data.
        """

        if pair == 'USDT/BTC':
            return self.get_usdtbtc_candles(exchange, size,
                                            start_time, end_time)
        
        query = f"""select \
        first(open) AS open, last(close) AS close, max(high) AS high, min(low)\
        as low, sum(volume) as volume from candles where exchange=$exchange\
        and pair=$pair and time>=$start_time and time<$end_time GROUP BY time({size}m)""" 

        bind_params = {
            'exchange': exchange,
            'pair': pair,
            'start_time': f"{start_time}".split('+')[0],
            'end_time': f"{end_time}".split('+')[0]

        }
        result = self.influx_client.query(
            query,
            bind_params=bind_params,
            database=self.db_name
        )
        candlesticks = result.get(self.TABLE_WITH_1MINUTE_CANDLES, pd.DataFrame())
        candlesticks['exchange'] = exchange
        candlesticks['pair'] = pair
        
        return candlesticks
    
    def get_usdtbtc_candles(self,
                            exchange: str, 
                            size: int,
                            start_time: datetime, 
                            end_time: datetime,
                            ) -> pd.DataFrame:
        
        df_candles_quote = self.get_candlestick('BTC/USDT', exchange, size,
                                                start_time, end_time)
        df_candles = pd.DataFrame(index = df_candles_quote.index,
                                  columns = ['open', 'high', 'low', 'close',
                                             'volume', 'close_time'])     
        df_candles[['open', 'close']] = 1/df_candles_quote[['open','close']] *10000
        df_candles.high = 1/df_candles_quote.low *10000
        df_candles.low = 1/df_candles_quote.high *10000
        df_candles.volume = df_candles_quote.volume
        return df_candles[['open','high','low','close','volume']]



MINUTES_TO_BINANCE_INTERVAL = {
    1: '1m',
    5: '5m',
    15: '15m',
    30: '30m',
    60: '1h',
    120: '2h',   
}

BINANCE_INTERVAL_TO_MINUTES = {
    MINUTES_TO_BINANCE_INTERVAL[k]: k for k in MINUTES_TO_BINANCE_INTERVAL
}

class BinancePricesRepository:

    def get_candles(self, base_currency, quote_currency,
                    interval, limit, end_time=None):
        print(base_currency, quote_currency, interval, limit, end_time)
        if base_currency == 'USDT' and quote_currency == 'BTC':
            df_candles = self._get_usdtbtc_candles(interval, limit, end_time)
        elif base_currency == quote_currency:  # both 'USDT' or both 'BTC'
            df_candles = self._get_dummy_candles(interval, limit, end_time)
        else:
            df_candles = self._get_candles(base_currency, quote_currency,
                                           interval, limit, end_time)
        return df_candles

    def _get_candles(self, base_currency, quote_currency, interval,
                     limit, end_time=None):
        candles = pd.DataFrame()
        current_end_time = end_time
        while len(candles) < limit: 
            current_limit = min(1000, limit - len(candles))
            older_candles = self._get_candles_core(base_currency,
                                                   quote_currency, interval,
                                                   current_limit, current_end_time)
            if len(older_candles) == 0:
                return candles
            candles = pd.concat([older_candles, candles])
            current_end_time = candles.index[0] - timedelta(minutes=1)
        return candles

    def _get_candles_core(self, base_currency, quote_currency, interval,
                          limit, end_time=None):
        symbol = f"{base_currency}{quote_currency}"

        if end_time is None:
            end_time=datetime.now()            
        _end_time = int(end_time.timestamp()) * 1000

        target_url = "https://api.binance.com/api/v3/klines?symbol={}&interval={}&limit={}&endTime={}".format(symbol, interval, limit, _end_time)
        response = requests.get(target_url)
        df = pd.DataFrame(response.json(),
                          columns=['open_time', 'open', 
                                    'high', 'low', 'close',
                                    'volume', 'close_time',
                                    'quote_asset_volume',
                                    'number_of_trades',
                                    'taker_buy_base_asset_volume',
                                    'taker_buy_quote_asset_volume',
                                    'ignore'])
        df.open_time = pd.to_datetime(df.open_time, unit="ms")
        df.close_time = pd.to_datetime(df.close_time, unit="ms")

        df.open = pd.to_numeric(df.open)
        df.close = pd.to_numeric(df.close)
        df.high = pd.to_numeric(df.high)
        df.low = pd.to_numeric(df.low)    
        df.volume = pd.to_numeric(df.volume)

        df.index = df.open_time

        return df[['open', 'high', 'low', 'close', 'volume', 'close_time']]

    def _get_usdtbtc_candles(self, interval, limit, end_time):
        df_candles_quote = self._get_candles('BTC', 'USDT', interval, 
                                             limit, end_time)
        df_candles = pd.DataFrame(index = df_candles_quote.index, 
                                  columns = ['open', 'high', 'low', 'close',
                                             'volume', 'close_time'])     
        df_candles[['open', 'close']] = 1/df_candles_quote[['open','close']]
        df_candles.high = 1/df_candles_quote.low
        df_candles.low = 1/df_candles_quote.high
        df_candles.volume = df_candles_quote.volume

        return df_candles[['open','high','low','close','volume']]

    def _get_dummy_candles(self, interval, limit, end_time):
        """
        Dummies candles are needed for USDT/USDT and BTC/BTC pairs 
        on holding strategies
        """
        df_candles_quote = self._get_candles('BTC', 'USDT', interval, 
                                             limit, end_time)
        df_candles = pd.DataFrame(index = df_candles_quote.index,
                                  columns = ['open', 'high', 'low', 'close',
                                             'volume', 'close_time'])     
        df_candles.open = 1
        df_candles.close = 1
        df_candles.high = 1
        df_candles.low = 1 
        df_candles.volume = 1

        return df_candles