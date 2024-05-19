import pandas as pd
import numpy as np 


def portfolio_returns(returns_df, weights, rate):
  """
  Function for calculating fee adjusted and weighted portfolios' returns
  ----------------------------------------------------------------------
  Inputs: 
    returns_df (dataframe) : return data of the assets without na row
    weights (array) : weight allocation of the assets
    rate (float) : transaction fee in percentage
  Return: 
    Portfolio Returns from Weight allocation
  """
  returns_df = returns_df.iloc[1:, :]
  return (
      (returns_df * weights[1:-1, :]).sum(axis = 1) - rate * np.absolute(weights[1:-1, :] - weights[:-2, :]).sum(axis = 1)
  )

def sortino_ratio_df(returns_df): 
  """
  Funcition for calculating sortino ratio of portfolios
  ----------------------------------------------------
  Inputs: 
    returns_df (dataframe) : return data of the assets without na row
  Returns: 
    pandas Series of Sortino Ratio of each asset
  """
  sortino_ratios = [] 
  for col in returns_df.columns: 
    downside_returns = returns_df[returns_df[col] < 0][col]
    sortino_ratios.append(
      returns_df[col].mean() / downside_returns.std()
    )
  return pd.Series(sortino_ratios, index = returns_df.columns)

def max_drawdown_df(returns_df): 
  """
  Inputs: 
    returns_df (dataframe) : return data of the assets without na row
  Returns: 
    pandas Series of Maximum DrawDown of each asset
    
  """
  return (returns_df.cummax() - returns_df.cumsum()).describe().loc['max']

if __name__ == '__main__': 
  pass
