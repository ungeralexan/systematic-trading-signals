
import numpy as np
import pandas as pd
from yahooquery import Ticker


# Data preparation


def download_stock_price_data(tickers, start_date, end_date):
    """
    Downloads close prices and compute multiplicative daily price changes.

    The function first tries to use Yahoo's adjusted close prices, because
    adjusted prices account for dividends and stock splits.  If adjusted
    close prices are not available, it falls back to ordinary close prices.

    Parameters
    ----------
    tickers : list
        Ticker symbols in the order that should appear in the output.
    start_date : str
        First date requested from Yahoo Finance
    end_date : str
        Last date requested from Yahoo Finance

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        First element: price level series with dates as index and tickers as
        columns.  
    """
    raw = Ticker(tickers).history(start=start_date, end=end_date)
    close_prices = raw["close"].unstack(level=0)

    if "adjclose" in raw.columns:
        adjusted_prices = raw["adjclose"].unstack(level=0)
        df_prices = adjusted_prices.where(~adjusted_prices.isna(),
                                          close_prices)
    else:
        df_prices = close_prices

    df_prices = df_prices[tickers].dropna()
    df_prices.index = pd.to_datetime(df_prices.index)

    return df_prices, price_changes_from_price_frame(df_prices)


def price_changes_from_price_frame(df_prices):
    """
    Convert a price DataFrame into multiplicative one-period changes.

    For every ticker and date after the first observation, the value is
    computed as price_t / price_{t-1}.  The first row is set to 1.0 because
    no earlier price is available.

    Parameters
    ----------
    df_prices : pandas.DataFrame
        Price levels with dates as index and tickers as columns.

    Returns
    -------
    pandas.DataFrame
        Multiplicative price changes with the same index and columns.
    """
    prices = df_prices.to_numpy(dtype=float)

    price_changes = np.ones(prices.shape)
    price_changes[1:, :] = prices[1:, :] / prices[:-1, :]

    df_price_changes = df_prices.copy(deep=True)
    df_price_changes[:] = price_changes

    return df_price_changes



# Shared numerical indicators


def moving_average(price_series, window_length):
    """
    Compute a trailing simple moving average.

    The value at date t uses only observations from t and earlier.  The first
    window_length - 1 observations are set to NaN because a full
    historical window is not yet available.

    Parameters
    ----------
    price_series : pandas.Series
        Price series indexed by date.
    window_length : int
        Number of observations in the trailing average.

    Returns
    -------
    pandas.Series
        Moving-average series with the same index as price_series.
    """
    prices = price_series.to_numpy(dtype=float)

    average = np.full(len(prices), np.nan)

    for index in range(window_length - 1, len(prices)):
        average[index] = np.sum(
            prices[index - window_length + 1:index + 1]
        ) / window_length

    return pd.Series(average, index=price_series.index)


def price_momentum(price_series, lookback_window):
    """
    Computes percentage price momentum over a fixed lookback window.

    Momentum is defined as price_t / price_{t-lookback_window} - 1. 

    Parameters
    ----------
    price_series : pandas.Series
        Price series indexed by date.
    lookback_window : int
        Number of periods used for the historical comparison.

    Returns
    -------
    pandas.Series
        Momentum series with NaN values before the first full lookback
        window is available.
    """
    prices = price_series.to_numpy(dtype=float)

    momentum = np.full(len(prices), np.nan)
    momentum[lookback_window:] = (
        prices[lookback_window:] / prices[:-lookback_window] - 1.0
    )

    return pd.Series(momentum, index=price_series.index)



#Performance statistics

def portfolio_returns_from_values(portfolio_values):
    """
    Convert portfolio values into simple one-period returns.

    Parameters
    ----------
    portfolio_values : 
        Portfolio value path, for example starting at 1.0.

    Returns
    -------
    numpy.ndarray
        Simple returns computed as value_t / value_{t-1} - 1.
    """
    values = np.asarray(portfolio_values, dtype=float)
    return values[1:] / values[:-1] - 1.0


def annualized_return(portfolio_values, periods_per_year=250):
    """
    Compute the compound annualized return of a portfolio value path.

    The default assumes 250 trading periods per year.  If the total growth is
    not positive, the annualized return is not economically meaningful and the
    function returns NaN.
    """
    values = np.asarray(portfolio_values, dtype=float)

    total_growth = values[-1] / values[0]
    number_of_periods = len(values) - 1

    if total_growth <= 0.0 or number_of_periods <= 0:
        return np.nan

    return total_growth ** (periods_per_year / number_of_periods) - 1.0


def annualized_volatility(returns, periods_per_year=250):
    """
    Compute annualized volatility from one-period returns.

    The result is scaled by the square root of the number of periods per year.
    """
    returns = np.asarray(returns, dtype=float)

    mean_return = np.sum(returns) / len(returns)
    variance = np.sum((returns - mean_return) ** 2) / len(returns)

    return np.sqrt(variance * periods_per_year)


def sharpe_ratio(returns, periods_per_year=250):
    """
    Compute the annualized Sharpe ratio with zero risk-free rate.

    The project uses a zero risk-free rate, so the numerator is
    the annualized mean return.
    """
    returns = np.asarray(returns, dtype=float)
    volatility = annualized_volatility(returns, periods_per_year)

    if volatility == 0.0:
        return np.nan

    mean_return = np.sum(returns) / len(returns)

    return mean_return * periods_per_year / volatility


def maximum_drawdown(portfolio_values):
    """
    Compute the maximum drawdown of a portfolio value path.

    Drawdown measures the percentage loss from the previous running maximum.
    """
    values = np.asarray(portfolio_values, dtype=float)

    running_maximum = np.maximum.accumulate(values)
    drawdowns = values / running_maximum - 1.0

    return np.min(drawdowns)


def performance_statistics_from_values(portfolio_values, periods_per_year=250):
    """
    Summarize a portfolio value path with common performance statistics.

    Returns annualized return, annualized volatility, Sharpe ratio, and maximum
    drawdown in a dictionary that can be converted directly into a table.
    """
    values = np.asarray(portfolio_values, dtype=float)
    returns = portfolio_returns_from_values(values)

    return {
        "Annualized Return": annualized_return(values, periods_per_year),
        "Annualized Volatility": annualized_volatility(
            returns,
            periods_per_year,
        ),
        "Sharpe Ratio": sharpe_ratio(returns, periods_per_year),
        "Max Drawdown": maximum_drawdown(values),
    }


# ---------------------------------------------------------------------------
# Signal and execution helpers
# ---------------------------------------------------------------------------

def signal_dataframe(index, raw_signal, **extra_columns):
    """
    Create a standardized signal DataFrame.

    The function converts every positive raw signal to 1.0 and every
    non-positive or non-finite signal to 0.0. 

    Parameters
    ----------
    index : pandas.Index
        Date index for the output DataFrame.
    raw_signal : array-like
        Raw signal values before binary cleaning.


    Returns
    -------
    pandas.DataFrame
        DataFrame with signal, position_change, and extra diagnostics.
    """
    signal = np.asarray(raw_signal, dtype=float)
    signal = np.where(np.isfinite(signal), signal, 0.0)
    signal = np.where(signal > 0.0, 1.0, 0.0)

    position_change = np.zeros(len(signal))
    position_change[1:] = signal[1:] - signal[:-1]

    signal_frame = pd.DataFrame(index=index)
    signal_frame["signal"] = signal
    signal_frame["position_change"] = position_change

    for name, values in extra_columns.items():
        signal_frame[name] = values

    return signal_frame


def apply_execution_delay(signal_frame, delay_days=1):
    """
    Shift a signal forward to avoid same-day execution assumptions.

    A delay of one day means that a signal observed at date t can only affect
    the portfolio from date t+1 onward.  This reduces look-ahead bias in the
    backtest.
    """
    delayed_signal = (
        signal_frame["signal"]
        .shift(delay_days)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )

    delayed = signal_frame.copy(deep=True)
    delayed["signal"] = delayed_signal

    position_change = np.zeros(len(delayed_signal))
    position_change[1:] = delayed_signal[1:] - delayed_signal[:-1]

    delayed["position_change"] = position_change

    return delayed


def single_signal_portfolio_values(price_values, signal_values,
                                   initial_value=1.0):
    """
    Simulate portfolio values for one long-only binary signal.

    The strategy earns the asset return only when the previous day's signal is
    1.0.  When the previous day's signal is 0.0, the portfolio is assumed to
    stay in cash with zero return.
    """
    prices = np.asarray(price_values, dtype=float)
    signals = np.asarray(signal_values, dtype=float)

    portfolio_values = np.ones(len(prices)) * initial_value

    for index in range(1, len(prices)):
        asset_return = prices[index] / prices[index - 1] - 1.0
        strategy_return = signals[index - 1] * asset_return
        portfolio_values[index] = (
            portfolio_values[index - 1] * (1.0 + strategy_return)
        )

    return portfolio_values


# Trading signals

def vix_term_structure_signal(
    target_series,
    vix_series,
    vix3m_series,
    ratio_threshold,
    momentum_window,
    momentum_threshold,
):
    """
    Build a signal from VIX term structure and target-asset momentum.

    Economic idea: the strategy invests in the target asset when the VIX3M/VIX
    ratio is above the chosen threshold and the target asset has
    positive enough momentum.  A high term-structure ratio is interpreted as
a less stressed
    volatility environment, while momentum requires asset-specific strength.
    """
    target_series = target_series.dropna()

    vix = vix_series.reindex(target_series.index).to_numpy(dtype=float)
    vix3m = vix3m_series.reindex(target_series.index).to_numpy(dtype=float)

    vix_ratio = vix3m / vix
    target_momentum = price_momentum(
        target_series,
        momentum_window,
    ).to_numpy(dtype=float)

    signal = np.where(
        (vix_ratio > ratio_threshold)
        & (target_momentum > momentum_threshold),
        1.0,
        0.0,
    )

    signal[:momentum_window] = 0.0

    return signal_dataframe(
        target_series.index,
        signal,
        vix_ratio=vix_ratio,
        target_momentum=target_momentum,
    )


def oil_energy_relative_strength_signal(
    target_series,
    oil_series,
    market_series,
    trend_window,
    oil_momentum_window,
    oil_momentum_threshold,
    target_momentum_window,
    target_momentum_threshold,
    relative_strength_window,
    relative_strength_threshold,
):
    """
    Build an energy-sector signal using trend and relative strength.

    Economic idea: the strategy invests in the target energy asset only when
    the asset is in an uptrend, oil has sufficient momentum, the target itself
    has sufficient momentum, and the target outperforms the broad market on a
    relative-strength basis.
    """
    target_series = target_series.dropna()

    target_prices = target_series.to_numpy(dtype=float)
    oil = oil_series.reindex(target_series.index)
    market = market_series.reindex(target_series.index)

    trend_average = moving_average(
        target_series,
        trend_window,
    ).to_numpy(dtype=float)

    oil_momentum = price_momentum(
        oil,
        oil_momentum_window,
    ).to_numpy(dtype=float)

    target_momentum = price_momentum(
        target_series,
        target_momentum_window,
    ).to_numpy(dtype=float)

    relative_strength = target_series / market

    relative_strength_momentum = price_momentum(
        relative_strength,
        relative_strength_window,
    ).to_numpy(dtype=float)

    signal = np.where(
        (target_prices > trend_average)
        & (oil_momentum > oil_momentum_threshold)
        & (target_momentum > target_momentum_threshold)
        & (relative_strength_momentum > relative_strength_threshold),
        1.0,
        0.0,
    )

    warmup_period = max(
        trend_window,
        oil_momentum_window,
        target_momentum_window,
        relative_strength_window,
    )

    signal[:warmup_period] = 0.0

    return signal_dataframe(
        target_series.index,
        signal,
        trend_average=trend_average,
        oil_momentum=oil_momentum,
        target_momentum=target_momentum,
        relative_strength_momentum=relative_strength_momentum,
    )



def gold_safe_haven_signal(
    gld_series,
    tlt_series,
    trend_window,
    momentum_window,
    gld_momentum_threshold,
    tlt_momentum_threshold,
):
    """
    Build a gold safe-haven signal from trend and dual momentum.

    Economic idea: the strategy invests in GLD only when gold is above its
    moving average, gold momentum is above the chosen threshold, and TLT
    momentum is also above the chosen threshold.  The TLT filter is a
    confirmation signal for a risk-off environment in which safe-haven assets
    can be in demand.
    """
    gld_series = gld_series.dropna()

    tlt = tlt_series.reindex(gld_series.index)

    trend_average = moving_average(
        gld_series,
        trend_window,
    ).to_numpy(dtype=float)

    gld_momentum = price_momentum(
        gld_series,
        momentum_window,
    ).to_numpy(dtype=float)

    tlt_momentum = price_momentum(
        tlt,
        momentum_window,
    ).to_numpy(dtype=float)

    gld_prices = gld_series.to_numpy(dtype=float)

    signal = np.where(
        (gld_prices > trend_average)
        & (gld_momentum > gld_momentum_threshold)
        & (tlt_momentum > tlt_momentum_threshold),
        1.0,
        0.0,
    )

    signal[:max(trend_window, momentum_window)] = 0.0

    return signal_dataframe(
        gld_series.index,
        signal,
        trend_average=trend_average,
        gld_momentum=gld_momentum,
        tlt_momentum=tlt_momentum,
    )
