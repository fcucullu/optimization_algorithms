import requests 
from datetime import datetime, timedelta
import pandas as pd
import json
import numpy as np
from scipy import optimize

class BrainyRecommendationsRepository:
    CLASSIC_USDT = 'classic_USDT'
    PRO_USDT = 'pro_USDT'
    CLASSIC_BTC = 'classic_BTC'
    PRO_BTC = 'pro_BTC'
    
    
    def __init__(self, target_url='http://10.0.61.78/recommendations/'):
        """
        nonprod target_url='http://10.0.61.78/recommendations/'
        prod target_url='http://10.0.102.34/recommendations/'
        old target_url='http://10.0.52.20:8000/v1/api/recomendador/'
        
        con el ultimo target debe usarse el método get_recommendations_old
        """
        self.target_url = target_url
    
    def get_recommendations(self, profile:str):
        params = {'perfil_de_riesgo': profile}
        response = requests.get(self.target_url+profile, params=params)
        recomendations = response.json()
        recomendations = pd.DataFrame(recomendations)
        recomendations.timestamp = pd.to_datetime(recomendations.timestamp)
        recomendations.set_index('timestamp', inplace=True)
        recomendations = recomendations.iloc[:,:-1]
        return 100*recomendations
    
    def get_historical_recommendations(self, profile:str, start_date:datetime, end_date:datetime):
        recomendations = []
        istart_date = start_date

        while istart_date < end_date:
            iend_date = min(istart_date+timedelta(days=5), end_date)
            params = {'start_date':istart_date.strftime('%Y-%m-%d'),
                     'end_date':iend_date.strftime('%Y-%m-%d')}
            response = requests.get(self.target_url+profile, params=params)
            recomendations = recomendations+(response.json())
            istart_date = iend_date

        recomendations = pd.DataFrame(recomendations)
        recomendations.timestamp = pd.to_datetime(recomendations.timestamp)
        recomendations.set_index('timestamp', inplace=True)
        recomendations = recomendations.sort_index()
        recomendations = 100*recomendations
        recomendations['portfolio'] = profile

        return recomendations
    
    def get_recommendations_without_fixed_percentage(self, profile:str,
                                               quote="USDT",
                                               percentage=75.):
        """
        Debe definirse además del profile:
            - quote: quote currency
            - percentage: porcentaje fijo
        """
        recomendations = self.get_recommendations(profile)
        recomendations[quote] = (recomendations[quote]-percentage)
        recomendations = recomendations.apply(lambda x : (100*x)/recomendations.sum(axis=1))
        return recomendations
    
    def get_recommendations_old(self, profile:str):
        params = {'perfil_de_riesgo': profile}
        response = requests.get(self.target_url, params=params)
        recomendations = response.json()
        recomendations = pd.DataFrame(
            [self._format_recommendatation(r) for r in recomendations]
        )
        recomendations.fecha = pd.to_datetime(recomendations.fecha)
        recomendations.set_index('fecha', inplace=True)
        recomendations = self._order_columns(recomendations)
        return recomendations
    

    def _format_recommendatation(self, raw_recommendation):
        d = json.loads(raw_recommendation['recomendacion'])

        recomendation_percentages = [
            list(d['USDT'].values())[0],            
            list(d['BTC'].values())[0],
            list(d['ETH'].values())[0],
            list(d['LTC'].values())[0],
            list(d['BNB'].values())[0],
        ]

        try:
            usdt, btc, eth, ltc, bnb = self._round_and_sum_up_100(recomendation_percentages)
        except:
            print(f"Fail on recommendations.py -- Recomendation_percentages: {recomendation_percentages}")
            print(d)
            usdt, btc, eth, ltc, bnb = np.nan, np.nan, np.nan, np.nan, np.nan 
        
        return {
            'id': raw_recommendation['id'],
            'fecha': raw_recommendation['fecha'],
            'riesgo': raw_recommendation['riesgo'],
            'lecturas': raw_recommendation['lecturas'],
            'intervalo': raw_recommendation['intervalo'],
            'perfil_de_riesgo': raw_recommendation['perfil_de_riesgo'],
            'currency': raw_recommendation['currency'],
            'by_recommender': raw_recommendation['by_recommender'],
            'BTC': btc,
            'ETH': eth,
            'LTC': ltc,
            'BNB': bnb,
            'USDT': usdt,
        }
    
    def _order_columns(self, df):
        columns = [
            'id', 'USDT', 'BTC', 'ETH', 'LTC', 'BNB', 'perfil_de_riesgo', 
            'currency', 'riesgo', 'lecturas', 'intervalo', 'by_recommender'
        ]
        return df[columns]

    def _round_and_sum_up_100(self, percentages):
        ints_and_decimals = [[int(p * 10000), (p * 10000) % 1, i] for i, p in enumerate(percentages)]
        units_needed = 10000 - sum([tripla[0] for tripla in ints_and_decimals])
        ints_and_decimals.sort(key=lambda tripla: tripla[1]) # I use decimal to order
        for i in range(units_needed):
            ints_and_decimals[i][0] += 1
        ints_and_decimals.sort(key=lambda tripla: tripla[2]) # I comeback to the original order
        return [tripla[0] / 100 for tripla in ints_and_decimals]    


CLASSIC_USDT = BrainyRecommendationsRepository.CLASSIC_USDT
PRO_USDT = BrainyRecommendationsRepository.PRO_USDT
CLASSIC_BTC = BrainyRecommendationsRepository.CLASSIC_BTC
PRO_BTC = BrainyRecommendationsRepository.PRO_BTC


class MarkowitzRecommendation:
    NUMBER_OF_CURRENCIES = 5

    MODELS = {
        "20191115" : {
            CLASSIC_USDT: { 'n_candles': 18, 'risk': 20, 'slope': 1100, 'commission_factor':  9},
            PRO_USDT:     { 'n_candles':  9, 'risk':  2, 'slope': 1100, 'commission_factor':  9},
            CLASSIC_BTC:  { 'n_candles': 11, 'risk': 40, 'slope': 1600, 'commission_factor':  9},
            PRO_BTC:      { 'n_candles':  9, 'risk': 12, 'slope': 1500, 'commission_factor': 22},
        },
        "20200515" : {
            CLASSIC_USDT: { 'n_candles': 11, 'risk': 58.83, 'slope':  248.00, 'commission_factor':  0.90},
            PRO_USDT:     { 'n_candles':  2, 'risk': 17.60, 'slope': 3141.13, 'commission_factor':  0.40},
            CLASSIC_BTC:  { 'n_candles': 86, 'risk': 82.78, 'slope':   37.17, 'commission_factor':  0.00},
            PRO_BTC:      { 'n_candles': 98, 'risk': 31.72, 'slope':  290.44, 'commission_factor':  0.02},
        },
        "20200528" : {
            CLASSIC_USDT: { 'n_candles':  9, 'risk': 82.61, 'slope': 2814.33, 'commission_factor': 4.38},
            PRO_USDT:     { 'n_candles':  6, 'risk': 20.55, 'slope': 3182.53, 'commission_factor': 4.38},
            CLASSIC_BTC:  { 'n_candles': 91, 'risk': 99.74, 'slope':   29.06, 'commission_factor': 4.38},
            PRO_BTC:      { 'n_candles': 40, 'risk': 65.64, 'slope':  688.27, 'commission_factor': 4.38},
        }
        #
        # USDT: factorcom 3 lecturas 53 pendiente 273.6 riesgo 0.32
        # BTC: factorcom 0.31 lecturas 81 pendiente 146 riesgo 7.06

    }

    def __init__(self, n_candles: int, risk: float, slope: float, commission_factor: float):
        self.risk = risk  # "riesgo"
        self.n_candles = n_candles  # "lecturas"
        self.slope = slope  # "pendiente"
        self.commission_factor = commission_factor # "factorcom"

    @classmethod
    def get_available_models(cls):
        """ Prints availables models (`date_label` + `profile` + its markowitz constans). `date_label` + `profile` 
        can be used on function `create_model` 
        """
        rows = []
        for date_label in cls.MODELS:
            for profile in cls.MODELS[date_label]: 
                row = cls.MODELS[date_label][profile].copy()
                row['date_label'] = date_label
                row['profile'] = profile
                rows.append(row)
        available_models = pd.DataFrame(rows)
        available_models = available_models[['date_label', 'profile', 'n_candles', 'risk', 'slope', 'commission_factor']]           

        return available_models

    @classmethod
    def create_model(cls, date_label: str, profile: str):
        """ create a model based on `date_label` and `profile`. See function `get_available_models` in order to see 
        possible values for these variables
        """
        return cls(**cls.MODELS[date_label][profile])

    def get_recommendations(self, closed_prices, initial_guess=(.2, .2, .2, .2, .2)):
        """ generate markowits recommendations 
        """

        # Input Data
        currencies_returns = closed_prices.pct_change()
        recent_returns = ((0.5)*(closed_prices.shift(2) + closed_prices.shift(1)) - closed_prices) / closed_prices
        rolling_mean_returns = currencies_returns.rolling(self.n_candles).mean().dropna()
        covariances = currencies_returns.rolling(self.n_candles).cov()

        # Configurations Variables
        constraints = {
            'type': 'eq', 
            'fun': lambda x: np.sum(x[:self.NUMBER_OF_CURRENCIES]) - 1.0 # solutions sum up to 1. 
        }
        bounds = [(0, None) for _ in range(self.NUMBER_OF_CURRENCIES)]
        solutions = []
        # create recommendations
        current_portofolio_composition = initial_guess
        for idx in rolling_mean_returns.index:
            mean_return = np.array(rolling_mean_returns.loc[idx]) * (365*12)
            covariance = np.matrix(covariances.loc[idx]) * (365 * 12)
            recent_return = np.array(recent_returns.loc[idx]) 

            def portfolio_function(x):
                commission = (np.where((x-current_portofolio_composition) < 0, 0, (x-current_portofolio_composition))).sum()
                return float(
                    - np.dot(mean_return, x)
                    - self.slope * np.dot(recent_return, x)
                    + self.risk  * np.dot(x, np.dot(covariance, x).T)
                    + self.commission_factor * commission
                )
            
            minimization = optimize.minimize(
                portfolio_function, initial_guess, bounds=bounds, constraints=constraints, method="SLSQP"
            )
            solutions.append(minimization.x)
            current_portofolio_composition = minimization.x            

        # Transformamos el vector de soluciones a un DataFrame
        portofolio_compositions = pd.DataFrame(
            solutions,
            index=rolling_mean_returns.index,
            columns=rolling_mean_returns.columns
        )
        portofolio_compositions = portofolio_compositions.apply(
            lambda x: round(x,6) * 100
        )

        return portofolio_compositions
