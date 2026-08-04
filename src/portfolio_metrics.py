import pandas as pd
import numpy as np
from data_fetcher import get_portfolio_data
from returns import calculate_daily_returns

def calculate_expected_returns(returns: pd.DataFrame) -> pd.Series:
    """
    Calculate expected returns (mean of daily returns) for each ticker.

    Parameters:
        returns: DataFrame of daily returns, one column per ticker, indexed by Date.
    Returns:    
        Annual returns for each ticker
    """

    mean_daily = returns.mean()
    annual_returns = mean_daily * 252  
    return annual_returns

def calculate_covariance_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the covariance matrix of daily returns.

    Parameters:
        returns: DataFrame of daily returns, one column per ticker, indexed by Date.

    Returns:
        Covariance matrix of daily returns.
    """
    return returns.cov() * 252


def calculate_portfolio_performance(weights: np.ndarray, expected_returns: pd.Series, cov_matrix: pd.DataFrame) -> tuple[float, float]:
    """
    Calculate the expected return and volatility of a portfolio given weights, expected returns, and covariance matrix.
    
    Parameters:
        weights: Array of portfolio weights (should sum to 1).
        expected_returns: Series of expected returns for each asset.
        cov_matrix: Covariance matrix of asset returns.
    
    Returns:
        Tuple containing expected portfolio return and portfolio volatility.
    """
    portfolio_return = np.dot(weights, expected_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_volatility

    

if __name__ == "__main__":

    prices = get_portfolio_data(["AAPL", "MSFT"])
    daily_returns = calculate_daily_returns(prices)

    exp_returns = calculate_expected_returns(daily_returns)
    cov_matrix = calculate_covariance_matrix(daily_returns)

    weights = np.array([0.6, 0.4])
    portfolio_return, portfolio_volatility = calculate_portfolio_performance(weights, exp_returns, cov_matrix)  

    print("Expected annual returns:")
    print(exp_returns)
    print("\nAnnualized covariance matrix:")
    print(cov_matrix)

    print("Expected portfolio return: ")
    print(portfolio_return)
    print("Expected portfolio volatility: ")
    print(portfolio_volatility)
    