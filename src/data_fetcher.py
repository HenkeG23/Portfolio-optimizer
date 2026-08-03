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

if __name__ == "__main__":
    df = get_price_data("AAPL")
    print(df.head())
    print(df.shape)