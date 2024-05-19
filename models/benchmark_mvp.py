import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

# 
def portfolio_stats(weights, mean_returns, cov_matrix):
    """
    Function to calculate portfolio mean return and variance
    --------------------------------------------------------
    Input:
        weights (array) : weight allocation to the assets
        mean_returns (array) : mean of the returns of the assets
        cov_matrix (array) : covariance matrix of the returns of the assets
    Return:
        portfolio_return (float) : return of the weight allocated portfolio
        portfolio_variance (float) : variance of the weight allocated portfolio
    """
    portfolio_return = np.sum(mean_returns * weights)
    portfolio_variance = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_variance

# Function to generate random portfolios
def generate_random_portfolios(num_portfolios, mean_returns, cov_matrix, seed):
    """
    Function for simulating N of portfolios from randomly generated weights
    ----------------------------------------------------------------------
    Inputs: 
        num_portfolios (int) : number of portfolio to simulate
        mean_returns (array) : mean of the returns of the assets 
        cov_matrix (array) : covariance matrix of the returns of the assets 
        seed (int) : seeding for random number generator
    Return: 
        results (array) : array storing return, variance and sharpe ratio of generated portfolio
        weights_record (array) : array storing generated random weights
    """
    np.random.seed(seed)
    results = np.zeros((3, num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_return, portfolio_variance = portfolio_stats(weights, mean_returns, cov_matrix)
        results[0,i] = portfolio_return
        results[1,i] = portfolio_variance
        results[2,i] = portfolio_return / portfolio_variance
    return results, weights_record


def mvp_mod(data, num_portfolios=10000, seed = 42):
    """
    Function for getting portfolio weights of maximum sharpe
    -------------------------------------------------------
    Inputs: 
        data (dataframe) : price data of assets
        num_portfolios (int) : number of portfolio to simulate
        seed (int) : seeding for random number generator
    """
    # Calculate mean returns and covariance matrix
    mean_returns = data.pct_change().dropna().mean()
    cov_matrix = data.pct_change().dropna().cov()
    results, weights = generate_random_portfolios(num_portfolios, mean_returns, cov_matrix, seed)
    
    # Find portfolio with maximum Sharpe ratio
    max_sharpe_idx = np.argmax(results[2])
    optimal_weights = weights[max_sharpe_idx]

    return optimal_weights    

if __name__ == '__main__':
    pass
    
    
