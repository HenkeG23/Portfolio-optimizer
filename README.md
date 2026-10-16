# Portfolio Optimizer

An interactive tool for portfolio optimization based on Modern Portfolio Theory (Markowitz). Given a set of stocks/ETFs, it finds the allocation that maximizes risk-adjusted return (Sharpe ratio) or minimizes volatility, and lets you compare these against your own weighting.

**[Open the live app on Streamlit Cloud](https://portfolio-optimizer-erbhuxdcgryfysc8gp3leh.streamlit.app/)**

![Portfolio Optimizer app](Assets/Screenshot.PNG)

## Why I Built This

As an active investor, I noticed that the number of holdings in my portfolio kept growing. The problem is that the more stocks you hold, the more the portfolio starts to look like the index, and if it ends up behaving like the index, you could just as well buy an index fund. Beating the index requires being selective.

I built this tool to support that process:

1. **Find holdings that don't pull their weight.** If the optimizer gives a stock zero weight, or removing it barely changes the portfolio's risk-adjusted return, it is a candidate to sell.
2. **Reduce the number of holdings in a structured way.** If two stocks have similar expected return, keep the one that adds less risk to the portfolio. This is not always the one with the lowest volatility on its own, since its correlation with the rest of the portfolio matters as much as its standalone risk.
3. **Quantify decisions.** Instead of relying on gut feeling, compare your current weights against the optimized portfolios and see the difference in return, volatility and Sharpe ratio.

## Features

**Data**
- Search for stocks/ETFs by company name (e.g. "volvo") or enter a ticker directly
- Historical prices from Yahoo Finance for any combination of markets
- All prices converted to a common currency (SEK, USD or EUR) using daily exchange rates, so currency risk is part of the analysis
- Different exchange holidays handled by carrying the last price forward instead of dropping days
- Analysis period of 1, 3, 5 or 10 years, or a custom date range

**Optimization**
- Maximum Sharpe ratio and minimum volatility portfolios via numerical optimization (scipy SLSQP)
- Optional max weight per stock, to avoid portfolios concentrated in one or two assets
- Analytical efficient frontier, shown together with a cloud of random portfolios

**Comparison**
- Manual portfolio weights (0.1% precision) compared side by side with the optimized portfolios
- Historical buy-and-hold performance of each portfolio against a benchmark (OMXS30 or S&P 500)

**Engineering**
- Local CSV caching and Streamlit caching to reduce API calls and keep the UI responsive
- Error handling for invalid tickers, empty inputs, infeasible weight caps and failed optimizations

## How It Works

Given historical returns and the covariance between assets, the tool solves for the weights that either maximize return per unit of risk (Sharpe ratio) or minimize total portfolio risk. Weights must sum to 1, and short selling is not allowed.

Two approaches are used side by side:

- **Monte Carlo simulation** generates thousands of random portfolios to visualize the risk/return space.
- **`scipy.optimize` (SLSQP)** solves for the exact optimal weights. This becomes clearly more accurate than random sampling as the number of assets grows, since random points cover a shrinking fraction of the possible weight combinations in higher dimensions.

The efficient frontier is computed by minimizing volatility for a range of target returns, starting from the minimum volatility portfolio. Portfolios below that point are inefficient, since a higher return is available for the same risk.

The historical performance chart shows how 10,000 (in the chosen currency) invested on the first day would have developed in each portfolio.

## Limitations

This is a textbook mean-variance implementation, and it inherits the known weaknesses of that model:

- **Estimation error in expected returns.** Historical mean returns are very noisy estimates of future returns. The max Sharpe optimizer is highly sensitive to small changes in these inputs and tends to concentrate weight in whatever performed best in the sample, effectively maximizing estimation error. The weight cap is a practical guard against this, not a solution.
- **Unstable covariance matrix.** The sample covariance becomes unreliable as the number of assets grows relative to the number of observations.
- **Volatility as the risk measure.** Standard deviation is symmetric: it penalizes upside moves as much as downside moves and does not capture tail risk.
- **In-sample backtest.** The historical performance chart uses weights optimized on the same period, so it overstates what could have been achieved in practice.
- **Forward-filled prices.** When markets in different countries have different holidays, the closed market's price is carried forward. This creates artificial zero-return days that bias estimated volatility and correlations downward.
- **Constant risk-free rate.** The same rate is used regardless of the chosen currency, although SEK, USD and EUR rates differ.

## Next Steps

- **Currency attribution.** Split each holding's return into the stock's own return and the exchange rate effect. For a Swedish investor with many US stocks, the USD/SEK rate can be a large part of the total return, and it is worth knowing how much.
- **Shrinkage estimation of the covariance matrix** (Ledoit-Wolf) to stabilize the optimization.
- **CVaR (Expected Shortfall)** as an alternative, tail-focused risk measure.
- **Walk-forward backtesting.** Optimize on a rolling window and evaluate on the following out-of-sample period.

## Tech Stack

- **Python**
- **Streamlit**: interactive UI
- **yfinance**: historical price data
- **scipy**: numerical optimization (SLSQP)
- **pandas / numpy**: data processing and linear algebra
- **matplotlib**: efficient frontier plot

## Project Structure

```
portfolio-optimizer/
├── app.py                    # Streamlit UI, entry point
├── requirements.txt
├── data/                     # Cached prices and currencies (gitignored)
└── src/
    ├── data_fetcher.py       # Data fetching, caching, ticker search, currency conversion
    ├── returns.py            # Prices → daily returns
    ├── portfolio_metrics.py  # Expected returns, covariance, portfolio performance, backtest
    ├── monte_carlo.py        # Random portfolios for visualization
    ├── optimizer.py          # Max Sharpe, min volatility, efficient frontier
    └── visualization.py      # Efficient frontier plot
```

## Running Locally

```powershell
git clone https://github.com/HenkeG23/portfolio-optimizer.git
cd portfolio-optimizer
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## What I Learned

- **yfinance's MultiIndex columns** silently break CSV caching if not flattened first. A good reminder that "it works on the first run" doesn't mean the pipeline is correct end to end.
- **Streamlit reruns the entire script on every interaction**, so `st.button` alone doesn't persist state. `st.session_state` is needed to keep results visible while the user adjusts other inputs, such as the manual weight sliders.
- **Monte Carlo simulation is intuitive but doesn't scale.** With more assets, the gap between random sampling and true optimization grows quickly.
- **Separation of concerns pays off.** Keeping data fetching, calculations, optimization and visualization in separate modules made debugging much easier, since each bug could be isolated to one small file.

## Author

Henrik Gidlund