import matplotlib.pyplot as plt
from utils.simulations import PriceWorker
import pandas as pd
import numpy as np
import pylab 
from scipy.stats import probplot
import seaborn as sns
import statsmodels.api as sm

class Graphs():
    '''En esta clase estan todas las funciones que grafican la información'''
    
    def graph_price_time(self, serie_price, label:str):
        fig=plt.figure(figsize=(12, 7))
        fig.suptitle(label, fontsize=22, fontweight="bold")
        plt.plot(serie_price, label='Simulation', lw=3, alpha=0.8)
        plt.xlabel('Time', fontsize=15)
        plt.ylabel('Price', fontsize=15)
        plt.show()
        
    def graph_two_price_time(self, serie_price_1, label_1:str, serie_price_2, label_2:str):
        fig=plt.figure(figsize=(12, 7))
        fig.suptitle("Comparison", fontsize=22, fontweight="bold")
        plt.plot(serie_price_1, label=label_1, lw=3, alpha=0.8)
        plt.plot(serie_price_2, label=label_2, lw=3 , alpha=0.8)
        plt.xlabel('Time', fontsize=15)
        plt.ylabel('Price', fontsize=15)
        plt.show()

    def graph_return_time(self, serie_price, label:str):
        returns = PriceWorker.calculate_logaritmic_returns(self, serie_price)
        fig=plt.figure(figsize=(12, 7))
        fig.suptitle(label, fontsize=22, fontweight="bold")
        plt.plot(returns, label='Simulation', lw=3, alpha=0.8)
        plt.xlabel('Time', fontsize=15)
        plt.ylabel('Returns', fontsize=15)
        plt.show()
    
    def graph_return_time_with_3std(self, serie_price, label:str):
        '''Identificacion de outliers segun Cartea y Figueroa (2005)'''
        tabla = pd.DataFrame()
        tabla['returns'] = PriceWorker.calculate_logaritmic_returns(self, serie_price)
        tabla['mean_return'] = tabla['returns'].mean()        
        tabla['3sd'] = 3*tabla['returns'].std()
        tabla['-3sd'] = -3*tabla['returns'].std()
        plt.figure(figsize=(12, 7))
        arr0 = plt.plot(tabla['returns'], lw=3, alpha=0.8)
        arr1 = plt.plot(tabla['mean_return'], lw=3, alpha=0.8)
        arr2 = plt.plot(tabla['3sd'], lw=3, alpha=0.8)
        arr3 = plt.plot(tabla['-3sd'], lw=3, alpha=0.8)
        plt.legend([arr0,arr1,arr2,arr3],['Retornos','Media','+3std','-3std'], loc=4, fontsize=12)
        plt.ylabel('Retornos logarítmicos ', fontsize=15)
        plt.xlabel('Tiempo', fontsize=15)
        plt.title(label, fontsize=22, fontweight="bold")
        plt.show()
        
    def graph_qqplot(self, serie_price, label:str):
        '''QQ Plot de la libreria scipy'''
        returns = PriceWorker.calculate_logaritmic_returns(self, serie_price)
        plt.figure(figsize=(12, 7))    
        probplot(returns, dist='norm', plot=pylab)
        plt.ylabel('Observaciones ordenadas', fontsize=15)
        plt.xlabel('Cuantiles teóricos', fontsize=15)
        plt.title(label, fontsize=22, fontweight="bold")
        pylab.show()

    def graph_comparison_pdf_returns_normal(self, serie_price, title):
        '''Grafica dos funciones de probabilidad (PDF)'''
        returns = PriceWorker.calculate_logaritmic_returns(self, serie_price)
        fig=plt.figure(figsize=(12, 7))
        fig.suptitle(title,fontsize=22, fontweight="bold")
        sns.distplot(returns, hist=False, kde=True, color='red', 
                       label='Retornos observados', kde_kws={'linewidth': 4})
        normal = np.random.normal(returns.mean(), returns.std(), len(returns))
        sns.distplot(normal, hist=False, kde=True, color='green', 
                     label='Normal', kde_kws={'linewidth': 4})
        plt.xlabel('Dominio de los retornos',fontsize=15)
        plt.ylabel('Densidad de probabilidad',fontsize=15)
        plt.legend(loc=1, fontsize=15)
        plt.ylim(-0.5, 12)
        plt.show()
    
    def graph_comparison_pdf_simulations(self, serie_price, simulation_price):
        '''Grafica dos funciones de probabilidad (PDF)'''
        returns = PriceWorker.calculate_logaritmic_returns(self, serie_price)
        simulated_returns = PriceWorker.calculate_logaritmic_returns(self, pd.DataFrame(simulation_price))
        fig=plt.figure(figsize=(12, 7))
        fig.suptitle('Comparación de densidades',fontsize=22, fontweight="bold")
        sns.distplot(returns, hist=False, kde=True, color='red', 
                       label='Retornos observados', kde_kws={'linewidth': 4})
        sns.distplot(simulated_returns, hist=False, kde=True, color='green', 
                     label='Simulacion', kde_kws={'linewidth': 4})
        plt.xlabel('Dominio de los retornos',fontsize=15)
        plt.ylabel('Densidad de probabilidad',fontsize=15)
        plt.legend(loc=1, fontsize=15)
        plt.show()    
        
    def graph_comparison_cdf_simulations(self, serie_price_1, label_1:str, serie_price_2, label_2:str):
        fig=plt.figure(figsize=(12, 7))
        fig.suptitle('Comparación a través del tiempo',fontsize=22, fontweight="bold")
        plt.plot(serie_price_1, label=label_1, lw=3, alpha=0.8)
        plt.plot(serie_price_2, label=label_2, lw=3, alpha=0.8)
        plt.xlabel('Tiempo', fontsize=15)
        plt.ylabel('Actuación acumulada', fontsize=15)
        plt.legend(loc=4, fontsize=15)
        plt.show()

    
    def graph_autocorrelation(self, serie_price):
        returns = PriceWorker.calculate_logaritmic_returns(self, serie_price)
        # autocorrelation
        fig = sm.graphics.tsa.plot_acf(returns, lags=40)
        plt.show()
        # partial autocorrelation
        fig = sm.graphics.tsa.plot_pacf(returns, lags=40)
        del fig
        plt.show()
        
    def graph_candlesticks(self, ohlc_data, name_diff=''):
        
        import plotly.graph_objects as go
        import plotly.offline as py_offline
        
        data = [ 
            go.Candlestick(
                x=ohlc_data.index,
                open=ohlc_data['open'+name_diff],
                high=ohlc_data['high'+name_diff],
                low=ohlc_data['low'+name_diff],
                close=ohlc_data['close'+name_diff]
                            )
                ]

        fig = go.Figure(data=data)
        
        py_offline.plot(fig, filename='Candle Stick')
        
    def graph_simple_poisson_proccess(self, title):
        np.random.seed(14)
        T = 1 
        N = 1000 #pasos temporales
        dt = T / N 
        lambdas = [1, 5, 10, 15] #Es la cantidad esperada de saltos por cada unidad de tiempo, SE CUMPLE!!
        X_T = [np.random.poisson(lam*dt, size=N) for lam in lambdas] #Proceso de Poisson propiamente dicho
        S = [[np.sum(X[0:i]) for i in range(N)] for X in X_T] #Acumulado del proceso
        X = np.linspace(0, 1, N)
        graphs = [plt.step(X, S[i], label="Lambda = %d"%lambdas[i])[0] for i in range(len(lambdas))]
        plt.legend(handles=graphs, loc=2, fontsize=15)
        plt.title(title, fontsize=22, fontweight="bold")
        plt.xlabel('Tiempo',fontsize=15)
        plt.ylabel('Saltos',fontsize=15)
        plt.ylim(0)
        plt.xlim(0)
        plt.show()
    
    def graph_compound_poisson_proccess(self, title):
        np.random.seed(11)
        T = 1 
        N = 1000 #pasos temporales
        dt = T / N 
        m = 0.5 #media del tamaño del salto
        delta = 1 #desvio estandar del tamaño del salto
        lambdas = [15] #Es la cantidad esperada de saltos por cada unidad de tiempo, SE CUMPLE!!
        X_T = [np.random.poisson(lam*dt, size=N) for lam in lambdas] #Proceso de Poisson propiamente dicho
        X = np.linspace(0, 1, N)
        normal = np.random.normal(m,delta,N)
        sumD = m * np.array(X_T) + delta * np.sqrt(X_T) * normal
        S = [[np.sum(X[0:i]) for i in range(N)] for X in sumD] #Acumulado del proceso
        graphs = [plt.step(X, S[i], label="Lambda = %d"%lambdas[i])[0] for i in range(len(lambdas))]
        plt.legend(handles=graphs, loc=2, fontsize=15)
        plt.title(title, fontsize=22, fontweight="bold")
        plt.xlabel('Tiempo',fontsize=15)
        plt.ylabel('Proceso Poisson compuesto',fontsize=15)
        plt.ylim(0)
        plt.xlim(0)
        plt.show()

    def check_simulation_correlation(self,data,list_simulations,savefig=False):
        simulations = list_simulations.copy()
        hdata = data.copy()
        for sim in simulations:
            sim["returns"]=sim["close"].pct_change()
            sim["upper"] = ((sim["high"] / np.maximum(sim["open"],sim["close"])) /
                (np.maximum(sim["open"],sim["close"]) / np.minimum(sim["open"],sim["close"]))) -1
            sim["lower"] = ((np.minimum(sim["open"],sim["close"]) / sim["low"]) /
             (np.maximum(sim["open"],sim["close"]) / np.minimum(sim["open"],sim["close"]))) -1
            sim["log_volume"] = np.log(sim["volume"])
        hdata["returns"]=hdata["close"].pct_change()
        hdata["upper"] = ((hdata["high"] / np.maximum(hdata["open"],hdata["close"])) /
            (np.maximum(hdata["open"],hdata["close"]) / np.minimum(hdata["open"],hdata["close"]))) -1
        hdata["lower"] = ((np.minimum(hdata["open"],hdata["close"]) / hdata["low"]) /
         (np.maximum(hdata["open"],hdata["close"]) / np.minimum(hdata["open"],hdata["close"]))) -1
        hdata["log_volume"] = np.log(hdata["volume"])
        l=1
        
        plt.figure(figsize=(20,12))
        for var in ["returns","upper","lower","log_volume"]:
            print(var)
            corr=[]
            for sim in simulations:
                corr.append(np.corrcoef(sim[var].shift(1)[2:],sim[var][2:])[1,0])
            corr_data=np.corrcoef(hdata[var].shift(1)[2:],hdata[var][2:])[1,0]
            print(np.mean(corr),corr_data)
            plt.subplot(2,2,l)   
            for sim in simulations:
                sim[var].hist(bins=50,density=1,histtype="step")
            hdata[var].hist(bins=50,density=1,histtype="step",color="black")
            plt.title(f"{var}\n corr t vs t-1, sim: {round(np.mean(corr),4)} data: {round(corr_data,4)}")
            
            l+=1
            
        if savefig:
            plt.savefig("check_simulation_correlations.pdf")
        plt.show()