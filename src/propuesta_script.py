from datetime import datetime
import pandas as pd
import numpy as np
import utils.graphs as gp
import utils.simulations as sim

prices = pd.read_csv(r'C:\Users\Francisco\Desktop\Facultad\Maestria en Finanzas Quantitativas\codigo_tesis\candles_4h.csv')
Graphs = gp.Graphs()


#Inputs
# start_date = datetime(2021,1,1)
# end_date = datetime(2022,1,1)
# candles_size_in_minutes = 60*24
# exchange = 'binance'
# pair = 'BTC/USDT'
# CandlestickRepository = utils.CandlestickRepository.default_repository()

# prices = CandlestickRepository.get_candlestick(pair,
#                                                exchange, 
#                                                candles_size_in_minutes,
#                                                start_date,
#                                                end_date)


returns = prices['open'].pct_change()
#por definicion 2:
logreturns = np.log(prices.close / prices.close.shift(1))

# Inputs for Brownian
p = prices.close
r = logreturns
S0 = p[0]
periods = 24/4*365
dt = 1 / periods
mu = logreturns.mean()
sigma = logreturns.std()

simulator_b = sim.GeometricBrownianMotionSimulator(S0, 32/36, mu, sigma)
brownian = pd.DataFrame(simulator.simulate(len(p)) )
r_brownian = np.log(brownian / brownian.shift(1))
Graphs.graph_comparison_brownian_merton_pdf(p, brownian, merton)

# Inputs for Merton
pw = sim.PriceWorker()
central_process_price, outliers_price = pw.filter_carterafigueroa(p)
muhat, sigmahat, Lambdahat, mu_jhat, sigma_jhat = \
    pw.calculate_parameters_mertonjumpdiffusion_aproximation(
        central_process_price, 
        outliers_price, periods)
simulator_m = sim.MertonJumpDiffusionSimulator(S0, dt, muhat, sigmahat, Lambdahat, mu_jhat, sigma_jhat)
merton = pd.DataFrame(simulator.simulate(len(p)) )
r_merton = np.log(merton / merton.shift(1))

###########################################################################
'''                     Grafico N°1                 ''' 

Graphs.graph_comparison_brownian_merton_pdf(p, brownian, merton)

###########################################################################
'''                     Grafico N°2                 ''' 

merton_series = simulator_m.simulate_n_times(5, len(p))
Graphs.graph_comparison_simulations(merton_series)


###########################################################################
'''                     Grafico N°4                 ''' 
Graphs.graph_comparison_two_normal(muhat, sigmahat, mu_jhat, sigma_jhat)



