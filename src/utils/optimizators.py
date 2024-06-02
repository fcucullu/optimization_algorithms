from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
import pandas as pd
from time import time

def bayes_optimize(strategy, 
                   bounds, 
                   acq_func, 
                   init_points=2, 
                   n_iter = 98, 
                   plot=False):
    
    init_time = time()
    objetive_function = strategy.get_performance    
    optimizer = BayesianOptimization(objetive_function,
                                       pbounds = bounds,
                                       random_state=0)
    optimizer.maximize(init_points=init_points,
                       n_iter=n_iter,
                       acq=acq_func)
    end_time = time()
    
    df = pd.DataFrame()
    df['strategy'] = [strategy.__class__.__name__]
    df['acq_func'] = [acq_func]
    df['performance'] = [optimizer.max['target']]
    df['params'] = [optimizer.max['params']]
    df['xratio'] = [strategy.get_xratio(*optimizer.max['params'].values())]
    df['delay'] = [end_time - init_time]

    if plot:
        print("Best result: {}; f(x) = {:.3f}.".format(optimizer.max["params"], optimizer.max["target"]))
        
        plt.figure(figsize = (15, 5))
        plt.plot(range(1, 1 + len(optimizer.space.target)), optimizer.space.target, "-o")
        plt.grid(True)
        plt.xlabel("Iteration", fontsize = 14)
        plt.ylabel("Performance f(x)", fontsize = 14)
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        plt.show()
    
    return df










