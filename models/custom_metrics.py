import pandas as pd
import numpy as np 


def portfolio_returns(returns, weights, rate):
  """
  
  """
  returns = returns.iloc[1:, :]
  return (
      (returns.iloc[1:, :] * weights[1:-2, :]).sum(axis = 1) - rate * np.absolute(weights[1:-2, :] - weights[:-3, :]).sum(axis = 1)
  )

def portfolio_values(price, weights):
  return (
      (price.iloc[1:, :] * (weights[:-1, :])).sum(axis = 1)
  )

def calculate_max_drawdown(prices):
  """
    Calculate the maximum drawdown for a time series of prices.
  
    Parameters:
    prices (list or numpy array): A list or array of asset prices.
  
    Returns:
    float: The maximum drawdown value.
  """
  prices = np.asarray(prices)
  # Calculate the cumulative maximum of the prices
  cum_max = np.maximum.accumulate(prices)
  # Calculate the drawdown
  drawdown = (prices - cum_max) / cum_max
  # Find the maximum drawdown
  max_drawdown = np.min(drawdown)
  
  return max_drawdown

if __name__ == '__main__': 
  pass
