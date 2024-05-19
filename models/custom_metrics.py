import pandas as pd
import numpy as np 


def portfolio_returns(returns, weights, rate):
  """
  
  """
  returns = returns.iloc[1:, :]
  return (
      (returns.iloc[1:, :] * weights[1:-2, :]).sum(axis = 1) - rate * np.absolute(weights[1:-2, :] - weights[:-3, :]).sum(axis = 1)
  )

def sortino_ratio_df(returns_df): 
  """
  """
  sortino_ratios = [] 
  for col in returns_df.columns: 
    downside_returns = returns_df[returns_df[col] < 0][col]
    sortino_ratios.append(
      returns_df[col].mean() / downside_returns.std()
    )
  return pd.Series(sortino_ratios, index = returns_df.columns)

def max_drawdown_df(returns_df): 
  return (returns_df.cummax() - returns_df.cumsum()).describe().loc['max']

if __name__ == '__main__': 
  pass
