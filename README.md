# Bayesian Optimization for Algorithmic Trading

> Gaussian Process-based hyperparameter optimization for trading strategy tuning. Solving the "black box" problem in quantitative finance.

## Problem

Algorithmic trading strategies have dozens of hyperparameters. Grid search is computationally expensive and doesn't scale. The objective function (strategy returns) is noisy, expensive to evaluate, and has no closed-form expression. You need a smarter way to find optimal configurations.

## Solution

Applied Bayesian optimization using Gaussian Processes to efficiently search the hyperparameter space:
- Models the objective function as a probabilistic surrogate
- Uses acquisition functions to balance exploration and exploitation
- Finds near-optimal configurations in far fewer evaluations than grid or random search

## Strategies Tested

Multiple trading strategies were optimized, including:
- VWAP vs SMA crossover
- Three Standard Moving Average
- Chandelier Exit
- Bollinger Bands Volume
- Various momentum and mean-reversion strategies

## Results

Results are exported as CSV files with Sharpe ratio, total returns, drawdown, and other performance metrics for each configuration tested.

```
Results/    # CSV outputs from optimization runs
src/        # Core optimization and strategy code
```

## Tech Stack

- Python, NumPy, SciPy
- Gaussian Process regression
- Custom backtesting framework
- Visualization and result analysis

## Author

**Francisco Cucullu** | [franciscocucullu.com](https://franciscocucullu.com)
