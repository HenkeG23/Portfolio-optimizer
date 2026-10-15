import json
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf

DATA_DIR = Path("data")
CACHE_MAX_AGE_HOURS = 24
CACHED_HISTORY_YEARS = 10

# Yahoo quotes some markets in minor currency units, e.g. London stocks in pence
MINOR_CURRENCIES = {
    "GBp": ("GBP", 100),
    "GBX": ("GBP", 100),
    "ZAc": ("ZAR", 100),
    "ILA": ("ILS", 100),
}


def years_ago(years: int) -> str:
    """Return the date `years` years back from today as "YYYY-MM-DD"."""
    return (datetime.now() - timedelta(days=years * 365)).strftime("%Y-%m-%d")


def get_price_data(ticker: str, start: str = None, end: str = None) -> pd.DataFrame:
    """
    Fetch historical price data for a single ticker.
    Uses local CSV cache if a fresh file exists, otherwise downloads from yfinance.
    The cache always holds at least CACHED_HISTORY_YEARS of history, so switching
    between shorter periods doesn't trigger new downloads.

    Parameters:
        ticker: Stock/ETF symbol, e.g. "AAPL"
        start: Start date as "YYYY-MM-DD". Defaults to 5 years ago.
        end: End date as "YYYY-MM-DD". Defaults to today.

    Returns:
        DataFrame with columns Open, High, Low, Close, Volume (indexed by Date).
    """
    if start is None:
        start = years_ago(5)
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    download_start = min(start, years_ago(CACHED_HISTORY_YEARS))

    DATA_DIR.mkdir(exist_ok=True)
    cache_file = DATA_DIR / f"{ticker}_{download_start}.csv"

    df = None
    if cache_file.exists():
        file_age_hours = (datetime.now().timestamp() - cache_file.stat().st_mtime) / 3600
        if file_age_hours < CACHE_MAX_AGE_HOURS:
            print(f"Loading {ticker} from cache...")
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)

    if df is None:
        print(f"Downloading {ticker} from yfinance...")
        df = yf.download(ticker, start=download_start, progress=False)

        if df.empty:
            raise ValueError(f"No data found for ticker '{ticker}'. Check the symbol and try again.")

        # Flatten MultiIndex columns (yfinance returns (Price, Ticker) pairs for single tickers too)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Remove older cache files for this ticker before writing the new one
        for old_file in DATA_DIR.glob(f"{ticker}_*.csv"):
            old_file.unlink()
        df.to_csv(cache_file)

    df = df.loc[start:end]
    if df.empty:
        raise ValueError(f"No data found for ticker '{ticker}' between {start} and {end}.")

    return df

def get_currency(ticker: str) -> str:
    """
    Look up the currency a ticker is quoted in, e.g. "USD" or "SEK".
    Results are cached in DATA_DIR/currencies.json since they rarely change.
    """
    cache_file = DATA_DIR / "currencies.json"
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}

    if ticker not in cache:
        try:
            currency = yf.Ticker(ticker).fast_info["currency"]
        except Exception:
            currency = None
        if not currency:
            raise ValueError(f"Could not determine the currency of '{ticker}'.")

        cache[ticker] = currency
        DATA_DIR.mkdir(exist_ok=True)
        cache_file.write_text(json.dumps(cache, indent=2))

    return cache[ticker]

def convert_currency(prices: pd.Series, from_currency: str, to_currency: str, start: str, end: str) -> pd.Series:
    """
    Convert a price series from one currency to another using daily exchange rates.

    Parameters:
        prices: Series of prices indexed by Date.
        from_currency: Currency the prices are quoted in, as reported by Yahoo (e.g. "USD", "GBp").
        to_currency: Target currency, e.g. "SEK".
        start: Start date as "YYYY-MM-DD".
        end: End date as "YYYY-MM-DD".

    Returns:
        Series of prices in the target currency.
    """
    currency, divisor = MINOR_CURRENCIES.get(from_currency, (from_currency, 1))
    prices = prices / divisor

    if currency == to_currency:
        return prices

    fx_rate = get_price_data(f"{currency}{to_currency}=X", start, end)["Close"]
    # Use the latest known rate for each trading day (FX and stock calendars differ)
    fx_rate = fx_rate.reindex(fx_rate.index.union(prices.index)).ffill().reindex(prices.index)

    return prices * fx_rate

def get_portfolio_data(tickers: list[str], start=None, end=None, base_currency: str = None):
    """
    Combine multiple tickers into a single DataFrame

    Parameters:
        tickers: List of stock/ETF symbols, e.g. ["AAPL", "MSFT"]
        start: Start date as "YYYY-MM-DD". Defaults to 5 years ago.
        end: End date as "YYYY-MM-DD". Defaults to today.
        base_currency: If given, all prices are converted to this currency, e.g. "SEK".

    Returns:
        DataFrame of daily closing prices, one column per ticker, indexed by Date.
    """
    if start is None:
        start = years_ago(5)
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    close_prices = {}
    for ticker in tickers:
        df = get_price_data(ticker, start, end)
        close = df["Close"]
        if base_currency is not None:
            close = convert_currency(close, get_currency(ticker), base_currency, start, end)
        close_prices[ticker] = close

    result = pd.DataFrame(close_prices)

    # Exchanges have different holidays: carry the last price forward instead of
    # dropping every day where any single market was closed
    result = result.ffill().dropna()

    return result

def search_tickers(query: str, max_results: int = 10) -> list[dict]:
    """
    Search Yahoo Finance for stocks/ETFs matching a company name or ticker.

    Parameters:
        query: Free text, e.g. "volvo" or "VOLV-B.ST"
        max_results: Maximum number of matches to return.

    Returns:
        List of dicts with keys symbol, name and exchange.
    """
    quotes = yf.Search(query, max_results=max_results, news_count=0).quotes

    matches = []
    for quote in quotes:
        if not quote.get("symbol") or quote.get("quoteType") not in ("EQUITY", "ETF"):
            continue
        matches.append({
            "symbol": quote["symbol"],
            "name": (quote.get("longname") or quote.get("shortname") or "").strip(),
            "exchange": quote.get("exchDisp", ""),
        })

    return matches


if __name__ == "__main__":
    df = get_portfolio_data(["AAPL", "MSFT"])
    print(df.head())
    print(df.shape)
