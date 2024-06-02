import pandas as pd
import numpy as np
from utils.candlestick import CandlestickRepository

def create_full_recommendation(
    portfolio_currency, 
    partial_recommendations, 
    partial_recommendations_percentage: int = 25, 
    ) -> pd.DataFrame:
    """ calculates the portfolio final compositions, this means, we calculate the portofolio
    compositions taking into account the `partial_recommendations` deals with
    `partial_recommendations_percentage` % of the total porfolio and the rest of it has to be 
    keeped in the `portolio_currency`.
    """
    
    recommendations = partial_recommendations.copy()
    recommendations = recommendations * (partial_recommendations_percentage / 100)
    recommendations[portfolio_currency] = recommendations[portfolio_currency] + (100 - partial_recommendations_percentage)
    
    return recommendations


class RecommendationsRunner:
    def run(self, portfolio_currency, recommendations: pd.DataFrame, prices_on_quote_currency: pd.DataFrame, initial_amount=100):
        """ calculate the returns of the portfolio that follows the recommendations minus commsions
        """
        shared_index = recommendations.index.intersection(prices_on_quote_currency.index)
        currency_returns = prices_on_quote_currency.loc[shared_index].ffill().replace(np.nan,1e-16).pct_change()
        weights = recommendations/100.
        final_weights = (weights.shift(1)*(1+currency_returns)).dropna() # muevo los pesos según los retornos
        final_weights = final_weights.apply(lambda x: x/final_weights.sum(axis=1)) # normalizo
        diff_weights = weights-final_weights # diferencia entre recomendación actual y nueva
        diff_weights = diff_weights.mul(np.where(diff_weights[portfolio_currency]>=0,1,-1),axis=0)# lo que necesito comprar o vender de USDT siempre es positivo
        comision_values = pd.DataFrame(0.001,index=diff_weights.index,columns=diff_weights.columns)#df con valores de comisiones
        comision_values[portfolio_currency] = 0.001 # comision USDT
        comision = (comision_values*diff_weights[diff_weights>0]).replace(np.nan,0).sum(axis=1) # comisión total
        returns = ((weights.shift(1)*currency_returns).dropna().sum(axis=1)-comision)
        returns[0]=0
        portfolio_evolution = weights.mul(initial_amount*(1+returns).cumprod(),axis=0)
        portfolio_evolution = portfolio_evolution/prices_on_quote_currency
        return portfolio_evolution, returns

    def run_old(self, portfolio_currency, recommendations: pd.DataFrame, prices_on_usdt: pd.DataFrame, initial_amount=100):
        """ calculate the returns of the portfolio that follows the recommendations minus commsions
        """
        current_portfolio = pd.Series(
            data = [0] * len(recommendations.columns), 
            index = recommendations.columns
        )
        current_portfolio[portfolio_currency] = initial_amount
        
        portfolio_evolution = []
        shared_index = recommendations.index.intersection(prices_on_usdt.index)
        for index, target_composition in recommendations.loc[shared_index].iterrows():
            portfolio_on_usdt = (prices_on_usdt.loc[index] * current_portfolio) 
            target_portfolio_on_usdt = (sum(portfolio_on_usdt) * target_composition/100)
            target_portfolio = target_portfolio_on_usdt / prices_on_usdt.loc[index]
            target_portfolio = target_portfolio.fillna(0)
            difference = target_portfolio - current_portfolio
            comission = difference.apply(lambda x: x * 0.001 if x > 0 else 0)
            target_portfolio = target_portfolio - comission
            
            portfolio_evolution.append(pd.DataFrame(data=[target_portfolio], index=[index]))
            current_portfolio = target_portfolio
        
        portfolio_evolution = pd.concat(portfolio_evolution)
        usdt_portfolio_value_evolution = (portfolio_evolution * prices_on_usdt).apply(sum, axis=1)
        returns = (usdt_portfolio_value_evolution / prices_on_usdt[portfolio_currency]).pct_change()
        return portfolio_evolution, returns
    
