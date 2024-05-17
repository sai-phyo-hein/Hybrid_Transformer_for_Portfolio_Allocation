import pandas as pd
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import duckdb as ddb
import datetime

# Function to calculate portfolio mean return and variance
def portfolio_stats(weights, mean_returns, cov_matrix):
    portfolio_return = np.sum(mean_returns * weights)
    portfolio_variance = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_variance

# Function to generate random portfolios
def generate_random_portfolios(num_portfolios, mean_returns, cov_matrix):
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


def mvp_mod(selected_tick, start_date, end_date, num_portfolios=10000):
    data = get_data(selected_tick, selected_col = 'Adj Close', start_date="2014-01-01", end_date="2024-03-31")
    
    # Calculate mean returns and covariance matrix
    mean_returns = data.pct_change().mean()
    cov_matrix = data.pct_change().cov()
    results, weights = generate_random_portfolios(num_portfolios, mean_returns, cov_matrix)
    
    # Find portfolio with maximum Sharpe ratio
    max_sharpe_idx = np.argmax(results[2])
    optimal_weights = weights[max_sharpe_idx]
    
    # Plot the efficient frontier
    print("*****************************************************************************************************************")
    print("*******************************Efficient Frontier & Maximum Sharpe Ratio Portfolio*******************************")
    plt.scatter(results[1,:], results[0,:], c=results[2,:], cmap='viridis')
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Volatility')
    plt.ylabel('Return')
    plt.scatter(results[1,max_sharpe_idx], results[0,max_sharpe_idx], marker='*', color='r', s=300, label='Max Sharpe Ratio')
    plt.legend()
    plt.title('Efficient Frontier')
    plt.show()
    
    
    print("******************************************************************************************************************")
    print("Weights for Maximum Sharpe Ratio Portfolio")
    print("Optimal Weights:", optimal_weights)
    print("******************************************************************************************************************")
    print("Expected Return of the Maximum Sharpe Ratio Portfolio")
    print(f"Expected Return of Portfolio : {portfolio_stats(optimal_weights, mean_returns, cov_matrix)[0]*100}")
    print("******************************************************************************************************************")
    print("Expected Standard Deviation of Maximum Sharpe Ratio Portfolio")
    print(f"Expected standard deviation of Portfolio : {portfolio_stats(optimal_weights, mean_returns, cov_matrix)[1]}")
    print("******************************************************************************************************************")
    
    return optimal_weights
    
    
    
    
    
