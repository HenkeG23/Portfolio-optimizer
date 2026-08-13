# Portfolio Optimizer

An interactive tool for portfolio optimization based on Modern Portfolio Theory (Markowitz). Given a set of stock/ETF tickers, it calculates the optimal asset allocation for maximizing risk-adjusted return (Sharpe ratio) or minimizing volatility, and lets users compare these against a manually weighted portfolio.

## Live Demo

[https://portfolio-optimizer-erbhuxdcgryfysc8gp3leh.streamlit.app/]

## Screenshot

![Portfolio Optimizer app](assets/screenshot.png)

## Features

- Fetch historical price data for any combination of stocks/ETFs via Yahoo Finance
- Calculate annualized expected returns and covariance matrix
- Find the portfolio with maximum Sharpe ratio using numerical optimization (scipy SLSQP)
- Find the minimum volatility portfolio
- Compute and visualize the analytical efficient frontier
- Manually set custom portfolio weights and compare against the optimized portfolios
- Side-by-side comparison table (Return, Volatility, Sharpe Ratio) across all three strategies
- Local CSV caching to reduce redundant API calls
- Error handling for invalid tickers and empty inputs

## How It Works

The tool implements Modern Portfolio Theory: given historical returns and the covariance between assets, it's possible to mathematically determine the asset weights that either maximize return per unit of risk (Sharpe ratio) or minimize total portfolio risk, subject to the constraint that weights sum to 1 and short-selling is not allowed.

Two complementary approaches are used. A Monte Carlo simulation generates thousands of random portfolios to visualize the risk/return space, while `scipy.optimize` (SLSQP method) solves for the exact optimal weights analytically — this becomes noticeably more accurate than Monte Carlo as the number of assets grows, since random sampling covers a shrinking fraction of the possible weight combinations in higher dimensions. The efficient frontier itself is computed by solving a constrained optimization problem (minimize volatility for a given target return) across a range of target returns.

## Tech Stack

- **Python**
- **Streamlit** - interactive UI
- **yfinance** - historical price data
- **scipy** - numerical optimization (SLSQP)
- **pandas / numpy** - data processing and linear algebra
- **matplotlib** - efficient frontier visualization

## Project Structure

```
portfolio-optimizer/
├── app.py                   # Streamlit UI - entry point
├── requirements.txt
├── data/                     # Cached price data (CSV, gitignored)
└── src/
    ├── data_fetcher.py       # Data fetching + CSV caching
    ├── returns.py             # Price → daily returns
    ├── portfolio_metrics.py   # Expected returns, covariance matrix, portfolio performance
    ├── monte_carlo.py         # Monte Carlo simulation (background visualization)
    ├── optimizer.py            # scipy.optimize: max Sharpe, min volatility, efficient frontier
    └── visualization.py       # matplotlib efficient frontier plot
```

## Running Locally

```powershell
git clone https://github.com/HenkeG23/Portfolio-optimizer.git
cd portfolio-optimizer
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## What I Learned

- **yfinance's MultiIndex columns** silently break CSV caching if not flattened first — a good reminder that "it works on the first run" doesn't mean the pipeline is correct end-to-end.
- **Streamlit reruns the entire script on every interaction**, which means `st.button` alone doesn't persist state — `st.session_state` is required to keep results visible while the user adjusts other inputs (e.g. manual weight sliders).
- **Monte Carlo simulation is a good visual/intuitive tool but doesn't scale well** — with more assets, the gap between random sampling and true analytical optimization (scipy's SLSQP) grows quickly, since random points cover a shrinking fraction of a higher-dimensional weight space.
- **Separation of concerns pays off in practice**: keeping data fetching, calculations, optimization, and visualization in separate modules made debugging significantly easier, since each bug could be isolated to a single, small file.

## Author

Henrik Gidlund