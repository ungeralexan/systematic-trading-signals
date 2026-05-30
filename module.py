"""
Reusable helper functions for the Computational Finance assessment.

The notebooks should contain the story, explanations, graphs, and final results.
This module should contain reusable code only: data download, signal functions,
and performance/statistics functions.

Important assessment hint:
Use NumPy for numerical computations. Pandas is used here mainly to keep the
same date index and column names as the input data.
"""

import numpy as np
import pandas as pd
from yahooquery import Ticker


# ---------------------------------------------------------------------------
# DATA DOWNLOAD
# ---------------------------------------------------------------------------

def download_stock_price_data(tickers, start_date, end_date):
    """
    Download adjusted close prices from Yahoo Finance and compute daily
    multiplicative price changes.

    Parameters
    ----------
    tickers : list[str]
        Example: ['AAPL', 'MSFT', 'AMZN', '^GSPC'].
    start_date : str
        Start date in format 'YYYY-MM-DD'.
    end_date : str
        End date in format 'YYYY-MM-DD'.

    Returns
    -------
    df_prices : pandas.DataFrame
        Adjusted close prices. Rows are dates, columns are tickers.
    df_price_changes : pandas.DataFrame
        Daily multiplicative changes. Example: 1.02 means +2%, 0.98 means -2%.
    """
    raw = Ticker(tickers).history(start=start_date, end=end_date)

    # Convert Yahoo output into a normal table: rows = dates, columns = tickers.
    df_prices = raw["adjclose"].unstack(level=0)
    df_prices = df_prices.dropna()

    prices = df_prices.to_numpy(dtype=float)

    # Compute price_t / price_{t-1}. The first row has no previous day, so we set
    # it to 1.0, meaning no change on the first observation.
    price_changes = np.ones_like(prices)
    price_changes[1:, :] = prices[1:, :] / prices[:-1, :]

    df_price_changes = pd.DataFrame(
        data=price_changes,
        index=df_prices.index,
        columns=df_prices.columns,
    )

    return df_prices, df_price_changes

def moving_average(values, window_length):
    """
    Compute a trailing moving average using only current and past values.

    This avoids data leakage: the value at time t only uses observations from
    t, t-1, ..., t-window_length+1.

    Parameters
    ----------
    values : array-like
        One-dimensional time series.
    window_length : int
        Number of observations used in the moving average.

    Returns
    -------
    averages : numpy.ndarray
        Moving average series. The first window_length - 1 entries are np.nan
        because there is not enough past data yet.
    """
    values = np.asarray(values, dtype=float)

    if window_length <= 0:
        raise ValueError("window_length must be positive.")

    if values.ndim != 1:
        raise ValueError("moving_average expects a one-dimensional array.")

    averages = np.full(values.shape[0], np.nan)

    if values.shape[0] < window_length:
        return averages

    # Cumulative sums allow us to compute rolling window sums efficiently.
    cumulative_sum = np.cumsum(np.insert(values, 0, 0.0))
    window_sums = cumulative_sum[window_length:] - cumulative_sum[:-window_length]
    averages[window_length - 1:] = window_sums / window_length

    return averages

####################################################
# Helping functions volatility filtered momentum signal
####################################################
def percentage_change(values, lookback_window):
    """
    Compute percentage change over a fixed lookback window.

    Example:
    If price today is 180 and price 120 days ago was 150,
    the result is 180 / 150 - 1 = 0.20.
    """
    values = np.asarray(values, dtype=float)

    if lookback_window <= 0:
        raise ValueError("lookback_window must be positive.")

    changes = np.full(values.shape[0], np.nan)

    if values.shape[0] <= lookback_window:
        return changes

    changes[lookback_window:] = (
        values[lookback_window:] / values[:-lookback_window] - 1.0
    )

    return changes

def rolling_standard_deviation(values, window_length):
    """
    Compute a trailing rolling standard deviation using NumPy.

    The value at time t only uses information from time t and earlier.
    """
    values = np.asarray(values, dtype=float)

    if window_length <= 0:
        raise ValueError("window_length must be positive.")

    rolling_std = np.full(values.shape[0], np.nan)

    if values.shape[0] < window_length:
        return rolling_std

    for end_index in range(window_length - 1, values.shape[0]):
        start_index = end_index - window_length + 1
        window = values[start_index:end_index + 1]
        rolling_std[end_index] = np.std(window)

    return rolling_std

def volatility_filtered_momentum_signal(
    series,
    momentum_window,
    volatility_window,
    volatility_threshold,
):
    """
    Volatility-filtered time-series momentum signal.

    Signal rule:
    signal_t = 1 if:
        price_t > price_{t-momentum_window}
        AND recent_volatility_t < volatility_threshold

    signal_t = 0 otherwise.

    Data science interpretation:
    This is a binary classifier based on two engineered features:
    1. lookback_return: trend feature
    2. recent_volatility: noise/stability feature
    """
    prices = series.to_numpy(dtype=float)

    # Daily simple returns, e.g. 0.02 means +2%.
    daily_returns = np.full(prices.shape[0], np.nan)
    daily_returns[1:] = prices[1:] / prices[:-1] - 1.0

    # Feature 1: momentum / past return
    lookback_return = percentage_change(prices, momentum_window)

    # Feature 2: recent volatility / recent noise
    recent_volatility = rolling_standard_deviation(
        daily_returns,
        volatility_window
    )

    signal = np.zeros(prices.shape[0])

    # Only compute signals when both features are available.
    valid_rows = ~np.isnan(lookback_return) & ~np.isnan(recent_volatility)

    positive_momentum = lookback_return > 0.0
    acceptable_volatility = recent_volatility < volatility_threshold

    signal[valid_rows] = np.where(
        positive_momentum[valid_rows] & acceptable_volatility[valid_rows],
        1.0,
        0.0,
    )

    position_change = np.diff(signal, prepend=0.0)

    signals = pd.DataFrame(index=series.index)
    signals["signal"] = signal
    signals["position_change"] = position_change
    signals["lookback_return"] = lookback_return
    signals["recent_volatility"] = recent_volatility

    return signals


####################################################
# Trading range breakout function 
####################################################
# Based on Brock, Lakonishok and LBaron (1992)
def trading_range_breakout_signal(series, breakout_window):
    """
    Long-only trading-range breakout signal.

    signal_t = 1 if price_t is greater than the maximum price over the
    previous breakout_window days.
    signal_t = 0 otherwise.

    Data science interpretation:
    This is a binary classifier using one engineered feature:
    whether today's price breaks above the recent historical maximum.
    """
    prices = series.to_numpy(dtype=float)

    if breakout_window <= 0:
        raise ValueError("breakout_window must be positive.")

    signal = np.zeros(prices.shape[0])
    recent_high = np.full(prices.shape[0], np.nan)

    for t in range(breakout_window, prices.shape[0]):
        past_window = prices[t - breakout_window:t]
        recent_high[t] = np.max(past_window)

        if prices[t] > recent_high[t]:
            signal[t] = 1.0
        else:
            signal[t] = 0.0

    position_change = np.diff(signal, prepend=0.0)

    signals = pd.DataFrame(index=series.index)
    signals["signal"] = signal
    signals["position_change"] = position_change
    signals["recent_high"] = recent_high
    signals["breakout_strength"] = prices / recent_high - 1.0

    return signals


###################################################################
# Simplidied long only short term reversal signal
###################################################################
def short_term_reversal_signal(
    series,
    lookback_window,
    return_threshold,
    holding_period=1,
):
    """
    Long-only short-term reversal signal.

    Signal idea:
    If the stock has fallen strongly over the last lookback_window days,
    the strategy invests for holding_period days. This is a long-only
    adaptation of the short-term reversal idea in Jegadeesh (1990) and
    Lehmann (1990), where recent losers tend to earn higher following
    returns than recent winners.

    Signal rule:
    lookback_return_t = price_t / price_{t-lookback_window} - 1

    entry_signal_t = 1 if lookback_return_t < return_threshold
    entry_signal_t = 0 otherwise

    signal_t = 1 for holding_period days after an entry signal
    signal_t = 0 otherwise

    Example:
    lookback_window = 5, return_threshold = -0.05, holding_period = 5
    means: if the stock lost more than 5% over the previous 5 trading days,
    invest for the next 5 signal observations.

    Parameters
    ----------
    series : pandas.Series
        Adjusted close price series.
    lookback_window : int
        Number of past observations used to compute the recent return.
    return_threshold : float
        Negative return threshold that triggers the reversal signal.
        Example: -0.05 means a loss larger than 5%.
    holding_period : int, default 1
        Number of observations for which the signal remains active after
        an entry signal.

    Returns
    -------
    signals : pandas.DataFrame
        DataFrame with columns:
        - signal: 1 if invested, 0 otherwise
        - position_change: change in the signal from the previous row
        - lookback_return: recent return used as the reversal feature
        - entry_signal: raw event indicator before applying holding period
    """
    prices = series.to_numpy(dtype=float)

    if lookback_window <= 0:
        raise ValueError("lookback_window must be positive.")

    if holding_period <= 0:
        raise ValueError("holding_period must be positive.")

    if return_threshold >= 0:
        raise ValueError("return_threshold should be negative for a reversal signal.")

    # Feature: recent return over the chosen lookback window.
    lookback_return = percentage_change(prices, lookback_window)

    entry_signal = np.zeros(prices.shape[0])
    valid_rows = ~np.isnan(lookback_return)

    # Event definition: recent loss is large enough to be classified as oversold.
    entry_signal[valid_rows] = np.where(
        lookback_return[valid_rows] < return_threshold,
        1.0,
        0.0,
    )

    signal = np.zeros(prices.shape[0])

    # Apply a fixed holding period after every entry signal.
    for start_index in range(prices.shape[0]):
        if entry_signal[start_index] == 1.0:
            end_index = min(start_index + holding_period, prices.shape[0])
            signal[start_index:end_index] = 1.0

    position_change = np.diff(signal, prepend=0.0)

    signals = pd.DataFrame(index=series.index)
    signals["signal"] = signal
    signals["position_change"] = position_change
    signals["lookback_return"] = lookback_return
    signals["entry_signal"] = entry_signal

    return signals




####################################################
# Moving average crossover signal
####################################################
# Based on Brock, Lakonishok and LeBaron (1992)
def ma_signal(series, short_window, long_window):
    """
    Moving-average crossover signal.

    Signal rule:
    signal_t = 1 if MA_t(short_window) > MA_t(long_window)
    signal_t = 0 otherwise

    This is the canonical trend-following rule studied in
    Brock, Lakonishok, and LeBaron (1992).

    Parameters
    ----------
    series : pandas.Series
        Adjusted close price series.
    short_window : int
        Window length for the short-term moving average.
    long_window : int
        Window length for the long-term moving average.

    Returns
    -------
    signals : pandas.DataFrame
        DataFrame with columns:
        - signal: 1 if short MA > long MA, 0 otherwise
        - position_change: +1 on entry, -1 on exit, 0 otherwise
        - ma_short: short-term moving average series
        - ma_long: long-term moving average series
    """
    if short_window >= long_window:
        raise ValueError("short_window must be smaller than long_window.")

    prices = series.to_numpy(dtype=float)

    ma_short = moving_average(prices, short_window)
    ma_long  = moving_average(prices, long_window)

    # Signal is only valid once both moving averages have enough history.
    signal = np.zeros(prices.shape[0])
    valid_rows = ~np.isnan(ma_short) & ~np.isnan(ma_long)

    signal[valid_rows] = np.where(
        ma_short[valid_rows] > ma_long[valid_rows],
        1.0,
        0.0,
    )

    position_change = np.diff(signal, prepend=0.0)

    signals = pd.DataFrame(index=series.index)
    signals["signal"]          = signal
    signals["position_change"] = position_change
    signals["ma_short"]        = ma_short
    signals["ma_long"]         = ma_long

    return signals


####################################################
# Performance and statistics functions
####################################################

def compute_portfolio_value(df_position):
    """
    Compute total portfolio value at each time step.

    Parameters
    ----------
    df_position : pandas.DataFrame
        Position DataFrame with stock columns + 'cash' column.

    Returns
    -------
    portfolio_value : numpy.ndarray
        Total portfolio value (stocks + cash) at each time step.
    """
    return df_position.sum(axis=1).to_numpy(dtype=float)


def compute_daily_returns(portfolio_value):
    """
    Compute daily simple returns from a portfolio value series.

    Parameters
    ----------
    portfolio_value : numpy.ndarray

    Returns
    -------
    daily_returns : numpy.ndarray
        Array of length len(portfolio_value) - 1.
    """
    return portfolio_value[1:] / portfolio_value[:-1] - 1.0


def annualized_return(daily_returns, trading_days=250):
    """
    Annualized mean return (arithmetic).
    """
    mean_daily = np.sum(daily_returns) / len(daily_returns)
    return mean_daily * trading_days


def annualized_volatility(daily_returns, trading_days=250):
    """
    Annualized volatility (standard deviation of daily returns).
    """
    mean_daily = np.sum(daily_returns) / len(daily_returns)
    variance   = np.sum(np.square(daily_returns - mean_daily)) / len(daily_returns)
    return np.sqrt(variance) * np.sqrt(trading_days)


def sharpe_ratio(daily_returns, risk_free_rate=0.0, trading_days=250):
    """
    Annualized Sharpe ratio.

    Parameters
    ----------
    daily_returns : numpy.ndarray
    risk_free_rate : float
        Annualized risk-free rate (e.g. 0.04 for 4%).
    trading_days : int

    Returns
    -------
    sharpe : float
    """
    ann_ret = annualized_return(daily_returns, trading_days)
    ann_vol = annualized_volatility(daily_returns, trading_days)
    if ann_vol == 0.0:
        return np.nan
    return (ann_ret - risk_free_rate) / ann_vol


def maximum_drawdown(portfolio_value):
    """
    Maximum drawdown: the largest peak-to-trough decline in portfolio value.

    Returns a value between 0 and 1 (e.g. 0.30 means -30%).

    Parameters
    ----------
    portfolio_value : numpy.ndarray

    Returns
    -------
    max_dd : float
    """
    running_max = np.maximum.accumulate(portfolio_value)
    drawdowns   = portfolio_value / running_max - 1.0
    return float(np.min(drawdowns))


def calmar_ratio(daily_returns, portfolio_value, trading_days=250):
    """
    Calmar ratio: annualized return divided by absolute maximum drawdown.

    A higher Calmar ratio means better return per unit of drawdown risk.
    """
    ann_ret = annualized_return(daily_returns, trading_days)
    max_dd  = maximum_drawdown(portfolio_value)
    if max_dd == 0.0:
        return np.nan
    return ann_ret / abs(max_dd)


def win_rate(daily_returns):
    """
    Fraction of trading days with a positive return.
    """
    return float(np.sum(daily_returns > 0.0) / len(daily_returns))


def compute_drawdown_series(portfolio_value):
    """
    Full drawdown time series (for plotting).

    Returns
    -------
    drawdown_series : numpy.ndarray
        Values between 0 and -1 at each time step.
    """
    running_max = np.maximum.accumulate(portfolio_value)
    return portfolio_value / running_max - 1.0


def compute_performance_table(label, daily_returns, portfolio_value, trading_days=250):
    """
    Collect all performance statistics into a dict for easy display.

    Parameters
    ----------
    label : str
        Name of the strategy or benchmark.
    daily_returns : numpy.ndarray
    portfolio_value : numpy.ndarray
    trading_days : int

    Returns
    -------
    stats : dict
    """
    return {
        "Strategy":             label,
        "Ann. Return":          annualized_return(daily_returns, trading_days),
        "Ann. Volatility":      annualized_volatility(daily_returns, trading_days),
        "Sharpe Ratio":         sharpe_ratio(daily_returns, trading_days=trading_days),
        "Max Drawdown":         maximum_drawdown(portfolio_value),
        "Calmar Ratio":         calmar_ratio(daily_returns, portfolio_value, trading_days),
        "Win Rate":             win_rate(daily_returns),
    }


if __name__ == "__main__":
    pass

