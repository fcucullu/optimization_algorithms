from utils.optimizators import bayes_optimize
import pandas as pd
import utils.simulations as sim


class Experiment:
    def __init__(self):
        return
    
    ACQUISITION_FUNCTIONS = {'upper confidece bound': 'ucb',
                             'expected improvement': 'ei',
                             'probability of improvement': 'poi'}
    
    def run_one_optimization_for_one_strategy(self, strategy, n_simulations, bounds=None):
        simulator = sim.SimulateMertonOHLC(strategy.candles, strategy.periods)
        simulations = simulator.simulate_ohlc_n_times(n_simulations, len(strategy.candles), volume=True)
        if bounds == None:
            bounds = strategy.get_standard_bounds()
        df = pd.DataFrame()
        df['best'] = 0
        for _, acq_func in self.ACQUISITION_FUNCTIONS.items():
            for simulated_data in simulations:
                strategy.candles = simulated_data        
                results = bayes_optimize(strategy, 
                                    bounds, 
                                    acq_func)
                df = df.append(results)
            df = df.reset_index(drop=True)
            df.loc[df.loc[df.acq_func == acq_func, 'xratio'].idxmax(),'best'] = 1
            df['week'] = strategy.candles.index[-1].isocalendar()[1]
        results = df.loc[df.best == 1].reset_index(drop=True)
        return results
    
    def divide_candles_in_weeks(self, candles, periods_in_weeks):
        list_of_periods = []
        candles['week'] = candles.index.isocalendar().week
        weeks_numbers = range(1,candles['week'][-1],periods_in_weeks)
        for week in weeks_numbers:
            end_date = candles.loc[candles.week == week].index[-1]
            start_date = end_date - pd.Timedelta(4, "w")
            list_of_periods.append([start_date, end_date])
        list_of_candles = []    
        for period in list_of_periods:
            list_of_candles.append(candles[period[0] : period[1]])
        return list_of_candles
    
    def run_multiple_optimizations_for_one_strategy(self, strategy, n_simulations, periods_in_weeks, bounds=None):
        candles = strategy.candles.copy()
        list_of_candles = Experiment.divide_candles_in_weeks(candles, periods_in_weeks)
        results = pd.DataFrame()
        for sample_candles in list_of_candles:
            strategy.candles = sample_candles.copy()
            df = Experiment.run_one_optimization_for_one_strategy(strategy, n_simulations, bounds)
            results = results.append(df)
        return results
    
    def run_multiple_optimizations_for_multiple_strategies(self, strategies, n_simulations, periods_in_weeks, bounds=None):
        general_results = pd.DataFrame()
        for strategy in strategies:
            results = self.run_multiple_optimizations_for_one_strategy(strategy, n_simulations, periods_in_weeks, bounds)
            general_results = general_results.append(results)
        return general_results
    

        
            