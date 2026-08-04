import pandas as pd
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

if __name__ == "__main__":

    prices = get_portfolio_data(["AAPL", "MSFT"])
    daily_returns = calculate_daily_returns(prices)

    exp_returns = calculate_expected_returns(daily_returns)
    cov_matrix = calculate_covariance_matrix(daily_returns)

    print("Expected annual returns:")
    print(exp_returns)
    print("\nAnnualized covariance matrix:")
    print(cov_matrix)