import pandas as pd
from data_fetcher import get_portfolio_data


def calculate_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily percentage returns from closing prices.

    Parameters:
        prices: DataFrame of closing prices, one column per ticker, indexed by Date.

    Returns:
        DataFrame of daily returns (same shape as input, minus first row).

    """
    returns = prices.pct_change().dropna()
    return returns

if __name__ == "__main__":
    prices = get_portfolio_data(["AAPL", "MSFT"])
    returns = calculate_daily_returns(prices)
    print(returns.head())
    print(returns.shape)