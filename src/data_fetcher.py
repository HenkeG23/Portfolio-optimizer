from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf

DATA_DIR = Path("data")
CACHE_MAX_AGE_HOURS = 24


def get_price_data(ticker: str, start: str = None, end: str = None) -> pd.DataFrame:
    """
    Fetch historical price data for a single ticker.
    Uses local CSV cache if a fresh file exists, otherwise downloads from yfinance.

    Parameters:
        ticker: Stock/ETF symbol, e.g. "AAPL"
        start: Start date as "YYYY-MM-DD". Defaults to 5 years ago.
        end: End date as "YYYY-MM-DD". Defaults to today.

    Returns:
        DataFrame with columns Open, High, Low, Close, Volume (indexed by Date).
    """
    if start is None:
        start = (datetime.now() - timedelta(days=5 * 365)).strftime("%Y-%m-%d")
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    DATA_DIR.mkdir(exist_ok=True)
    cache_file = DATA_DIR / f"{ticker}.csv"

    if cache_file.exists():
        file_age_hours = (datetime.now().timestamp() - cache_file.stat().st_mtime) / 3600
        if file_age_hours < CACHE_MAX_AGE_HOURS:
            print(f"Loading {ticker} from cache...")
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    print(f"Downloading {ticker} from yfinance...")
    df = yf.download(ticker, start=start, end=end)

    # Flatten MultiIndex columns (yfinance returns (Price, Ticker) pairs for single tickers too)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.to_csv(cache_file)
    return df

def get_portfolio_data(tickers: list[str], start=None, end=None):
    """
    Combine multilpe tickers into a single DataFrame

    Parameters:
        tickers: List of stock/ETF symbols, e.g. ["AAPL", "MSFT"]
        start: Start date as "YYYY-MM-DD". Defaults to 5 years ago.
        end: End date as "YYYY-MM-DD". Defaults to today.
    
    Returns:
        DataFrame of daily closing prices, one column per ticker, indexed by Date.
    """
    if start is None:
        start = (datetime.now() - timedelta(days=5 * 365)).strftime("%Y-%m-%d")
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    close_prices = {}
    for ticker in tickers:
        df = get_price_data(ticker, start, end)
        close_prices[ticker] = df["Close"]

    result = pd.DataFrame(close_prices)

    result = result.dropna()

    return result


if __name__ == "__main__":
    df = get_portfolio_data(["AAPL", "MSFT"])
    print(df.head())
    print(df.shape)