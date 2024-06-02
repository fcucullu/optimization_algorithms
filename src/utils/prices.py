import pandas as pd
from utils.candlestick import CandlestickRepository
from datetime import datetime

class PricesRepository:

    candlestick_label = 'open'
    candles_repository = CandlestickRepository.default_repository()

    def __init__(self, currency: str, start_time: datetime, end_time: datetime, currencies: list, n_minutes: int):
        self.currency = currency  
        self.start_time = start_time  
        self.end_time = end_time
        self.currencies = currencies
        self.n_minutes = n_minutes

    @classmethod
    def get_prices(cls, currency: str, start_time: datetime, end_time: datetime, currencies: list, n_minutes: int):
        obj = cls(currency, start_time, end_time, currencies, n_minutes)
        return obj._get_prices()

    def _get_prices(self):
        """Returns a dataframe with the prices for `currency` (constant 1) and `currencies` (prices
        based on `currency`) for period [start_time, end_time), using candlestick of sice `n_minutes` 
        and label `candlestick_label` (open, close, high, low)
        """
        
        prices = self._get_currencies_candlestick()
        prices = self._get_dataframe_of_prices_on_requested_currency(prices)
        prices = self._remove_na(prices)
        return prices

    def _get_currencies_candlestick(self):
        if self.end_time is None:
            end_time = datetime.utcnow()
        
        if self.currency == 'USDT':
            prices = self._get_currencies_candlestick_with_usdt_quote()
        else:
            prices = self._get_currencies_candlestick_with_btc_quote()
        return prices

    def _get_dataframe_of_prices_on_requested_currency(self, prices): 
        selected_prices = {} 

        if self.currency == 'USDT':
            selected_prices["USDT"]=1
        else:
            selected_prices['BTC']  = 1
            selected_prices['USDT'] = 1 / prices['BTC'][self.candlestick_label]
            
        for currency_ in self.currencies:
            if self.currency == currency_ or currency_ == 'USDT':
                continue
            selected_prices.update({f"{currency_}" : prices[f'{currency_}'][self.candlestick_label] })

        selected_prices=pd.DataFrame(data=selected_prices, index=prices[f"{self.currencies[0]}"].index)
        return selected_prices

    def _get_currencies_candlestick_with_usdt_quote(self):
        prices = {}
        for currency_ in self.currencies:
            prices[currency_] = self.candles_repository.get_candlestick(
                f'{currency_}/USDT', 'binance', self.n_minutes, self.start_time, self.end_time
            )  
        return prices 

    def _get_currencies_candlestick_with_btc_quote(self):
        prices = {}
        currencies_that_need_to_be_calculated = []

        for currency_ in self.currencies:
            if currency_ == self.currency:
                continue    
            prices[currency_] = self.candles_repository.get_candlestick(
                f'{currency_}/BTC', 'binance', self.n_minutes, self.start_time, self.end_time
            )
            if prices[currency_].empty:
                currencies_that_need_to_be_calculated.append(currency_)
                continue   
        
        prices_not_calculated = self._get_currencies_candlestick_not_calculated_with_btc_quote(
            currencies_that_need_to_be_calculated)
        
        prices.update(prices_not_calculated)
        
        return prices

    def _get_currencies_candlestick_not_calculated_with_btc_quote(self, currencies_not_calculated: list):
        prices = {}
        prices['BTC'] = self.candles_repository.get_candlestick(
                'BTC/USDT', 'binance', self.n_minutes, self.start_time, self.end_time
                )

        for currency_ in currencies_not_calculated:
            unconsistent_candles_sticks =self.candles_repository.get_candlestick(
                f'{currency_}/USDT', 'binance', self.n_minutes, self.start_time, self.end_time)._get_numeric_data() / prices['BTC']._get_numeric_data()
            consistent_data = unconsistent_candles_sticks[['open', 'close']]
            prices[currency_] = consistent_data

        return prices

    def _remove_na(self, prices):
        row_has_na = prices.apply(lambda row : row.isna().any(), axis = 1)
        if row_has_na.any():
            print("There are missing values. If you want to remove them apply dropna() to the output DataFrame")
            print(prices[row_has_na])
            #prices = prices[~row_has_na]
        return prices

