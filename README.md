# Systematic Trading Signals

> Empirical research and backtesting framework for rule-based equity trading strategies, implemented from scratch in Python and NumPy.

---

## Overview

This project implements and evaluates three systematic trading signals on US equities, combining classical financial econometrics with a clean, modular Python codebase. Each signal is grounded in peer-reviewed academic literature, calibrated through in-sample parameter optimisation, and validated on a held-out out-of-sample universe — following the same train/test discipline used in quantitative finance.

The work was developed as part of the *Computational Finance* course at **Universität Tübingen** (Summer Term 2026).

---

## Signals Implemented

| # | Signal | Type | Literature |
|---|--------|------|-----------|
| 1 | **Volatility-Filtered Momentum** | Trend-following | Moskowitz, Ooi & Pedersen (2012) |
| 2 | **Trading-Range Breakout** | Trend-following | Brock, Lakonishok & LeBaron (1992) |
| 3 | **Short-Term Reversal** | Contrarian | Jegadeesh (1990), Lehmann (1990) |

The three signals intentionally span two distinct economic mechanisms — trend-following and mean reversion — to provide strategy-level diversification.

---

## Key Results (Out-of-Sample, 2020–2024)

Evaluated on 6 stocks: AAPL, MSFT, AMZN (calibration) + GOOGL, META, NVDA (unseen).

| Strategy | Avg Sharpe | Avg Ann. Return | Avg Max Drawdown |
|----------|-----------|----------------|-----------------|
| S1 — Momentum | **0.92** | 17.5% | -18.8% |
| S2 — Breakout | 0.23 | 2.4% | **-11.8%** |
| S3 — Reversal | 0.59 | 15.8% | -39.9% |
| Buy & Hold | 0.89 | 35.2% | -52.0% |

Signal 1 achieves a higher average Sharpe than passive buy-and-hold (0.92 vs 0.89) while reducing maximum drawdown by more than half (-18.8% vs -52.0%).

---

## Repository Structure

```
systematic-trading-signals/
│
├── module.py                          # All reusable functions (signals, statistics, data)
├── assessment_notebook.ipynb          # Final assessment: 3 signals applied to a portfolio
├── research_notebook.ipynb            # Individual empirical research: parameter search,
│                                      #   in-sample / out-of-sample validation, comparison
└── moving_average_signal_explanation.tex  # LaTeX documentation of signal mathematics
```

---

## Module Architecture

All numerical computation lives in `module.py` and is importable in both notebooks. The module is organised into four layers:

```
module.py
├── Data
│   └── download_stock_price_data()        — Yahoo Finance via yahooquery
│
├── Signal Construction
│   ├── moving_average()                   — trailing MA, cumsum trick, O(n)
│   ├── ma_signal()                        — MA crossover signal (Brock et al. 1992)
│   ├── volatility_filtered_momentum_signal()  — momentum + vol filter (Moskowitz et al. 2012)
│   ├── trading_range_breakout_signal()    — resistance breakout (Brock et al. 1992)
│   └── short_term_reversal_signal()       — contrarian reversal (Jegadeesh 1990)
│
├── Feature Engineering Helpers
│   ├── percentage_change()                — lookback return over k days
│   └── rolling_standard_deviation()       — trailing volatility estimate
│
└── Performance & Statistics
    ├── compute_daily_returns()
    ├── annualized_return()
    ├── annualized_volatility()
    ├── sharpe_ratio()
    ├── maximum_drawdown()
    ├── calmar_ratio()
    ├── win_rate()
    ├── compute_drawdown_series()          — full drawdown time series for plotting
    └── compute_performance_table()        — consolidated stats dict
```

**Implementation constraint:** all numerical computations use NumPy only. No Pandas built-ins (`.rolling()`, `.mean()` etc.) are used for signal logic — every operation is implemented explicitly to ensure full transparency of the computational steps.

---

## Methodology

### Data
- Universe: AAPL, MSFT, AMZN, GOOGL, META, NVDA, ^GSPC
- Source: Yahoo Finance (adjusted close prices via `yahooquery`)
- Full period: 2012-05-18 — 2024-12-30 (3,174 trading days)
- Start date constrained by META IPO (May 2012)

### Train / Test Split
| Period | Dates | Days | Purpose |
|--------|-------|------|---------|
| In-sample | 2012-05-18 – 2019-12-31 | 1,917 | Parameter grid search |
| Out-of-sample | 2020-01-02 – 2024-12-30 | 1,257 | Held-out validation |

### Parameter Selection
Each signal is calibrated by grid search over economically motivated parameter ranges, ranked by average Sharpe ratio across the three calibration stocks (AAPL, MSFT, AMZN). Parameters are selected based on robustness across stocks, not peak performance on any single stock.

### Out-of-Sample Validation
Best in-sample parameters are applied without modification to:
1. The calibration stocks in the held-out 2020–2024 period
2. Three stocks never seen during calibration (GOOGL, META, NVDA)

---

## Performance Statistics Reference

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| Sharpe Ratio | Ann. Return / Ann. Volatility | Return per unit of risk. >1.0 is strong |
| Max Drawdown | min(V_t / max(V_s, s≤t) − 1) | Worst peak-to-trough loss |
| Calmar Ratio | Ann. Return / \|Max Drawdown\| | Return per unit of worst-case loss |
| Win Rate | % days with positive return | How often the strategy gains while invested |

---

## Academic References

```
Brock, W., Lakonishok, J., & LeBaron, B. (1992).
  Simple Technical Trading Rules and the Stochastic Properties of Stock Returns.
  The Journal of Finance, 47(5), 1731–1764.

Moskowitz, T. J., Ooi, Y. H., & Pedersen, L. H. (2012).
  Time Series Momentum.
  Journal of Financial Economics, 104(2), 228–250.

Jegadeesh, N. (1990).
  Evidence of Predictable Behavior of Security Returns.
  The Journal of Finance, 45(3), 881–898.

Lehmann, B. N. (1990).
  Fads, Martingales, and Market Efficiency.
  The Quarterly Journal of Economics, 105(1), 1–28.
```

---

## Roadmap

- [ ] Add RSI-based signal (momentum oscillator)
- [ ] Add portfolio-level backtesting across all three signals simultaneously
- [ ] Add transaction cost modelling (bid-ask spread, slippage)
- [ ] Extend universe to European equities (DAX, EURO STOXX 50)
- [ ] Add walk-forward optimisation to replace static train/test split
- [ ] Interactive visualisation dashboard (Plotly / Dash)

---


*Universität Tübingen · Computational Finance · Summer Term 2026*
