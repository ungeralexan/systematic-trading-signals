
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

def sortino_ratio(returns, periods_per_year=250):
    returns = np.asarray(returns, dtype=float)
    negative_returns = returns[returns < 0.0]
    mean_return = np.sum(returns) / len(returns)
    downside_variance = np.sum(negative_returns ** 2) / len(returns)
    downside_deviation = np.sqrt(downside_variance * periods_per_year)

    if downside_deviation == 0.0:
        return np.nan

    return mean_return * periods_per_year / downside_deviation

    
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
            periods_per_year,),
        "Sharpe Ratio": sharpe_ratio(returns, periods_per_year),
        "Sortino Ratio": sortino_ratio(returns, periods_per_year),
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
    signal = signal_frame["signal"].to_numpy(dtype=float)
    delayed_signal = np.zeros(len(signal))

    if delay_days <= 0:
        delayed_signal[:] = signal
    elif delay_days < len(signal):
        delayed_signal[delay_days:] = signal[:-delay_days]

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

    market_prices = market.to_numpy(dtype=float)
    relative_strength_values = target_prices / market_prices
    relative_strength = pd.Series(
        relative_strength_values,
        index=target_series.index,
    )

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


# Plotting helpers used by the assessment notebook

def drawdown_percent(curve):
    """
    Compute the percentage drawdown from the running historical peak.
    """
    values = np.asarray(curve, dtype=float)
    running_peak = np.maximum.accumulate(values)
    return (values - running_peak) / running_peak * 100.0


def rolling_sharpe_ratio(returns, window_length=250):
    """
    Compute a rolling annualized Sharpe ratio over a fixed return window.
    """
    returns = np.asarray(returns, dtype=float)
    rolling_values = np.full(len(returns), np.nan)

    for index in range(window_length, len(returns)):
        rolling_values[index] = sharpe_ratio(
            returns[index - window_length:index]
        )

    return rolling_values


def plot_drawdown_and_rolling_sharpe(
    portfolio_curve,
    benchmark_curve,
    dates,
    rolling_window=250,
):
    """
    Plot drawdown and rolling Sharpe ratio for the strategy and benchmark.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    dates = np.asarray(dates)
    portfolio_values = np.asarray(portfolio_curve, dtype=float)
    benchmark_values = np.asarray(benchmark_curve, dtype=float)

    portfolio_returns = portfolio_returns_from_values(portfolio_values)
    benchmark_returns = portfolio_returns_from_values(benchmark_values)
    return_dates = dates[1:]

    fig, (axes1, axes2) = plt.subplots(2, 1, figsize=(12, 12))

    axes1.fill_between(
        dates,
        drawdown_percent(portfolio_values),
        alpha=0.6,
        color="steelblue",
        label="Our Strategy",
    )
    axes1.fill_between(
        dates,
        drawdown_percent(benchmark_values),
        alpha=0.35,
        color="tomato",
        label="Benchmark AOR ETF",
    )
    axes1.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    axes1.set_title("Drawdown")
    axes1.set_ylabel("Drawdown (%)")
    axes1.legend()
    axes1.grid(alpha=0.4)
    plt.tight_layout()

    axes2.plot(
        return_dates,
        rolling_sharpe_ratio(portfolio_returns, rolling_window),
        color="steelblue",
        label="Our Strategy",
    )
    axes2.plot(
        return_dates,
        rolling_sharpe_ratio(benchmark_returns, rolling_window),
        color="tomato",
        linestyle="--",
        label="Benchmark AOR ETF",
    )
    axes2.axhline(0, color="black", linewidth=0.8, linestyle=":")
    axes2.axhline(1, color="green", linewidth=0.8, linestyle=":", alpha=0.6)
    axes2.set_title(f"Rolling Sharpe Ratio ({rolling_window}-Days)")
    axes2.legend()
    axes2.grid(alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_entry_exit_times(
    tickers,
    df_prices,
    df_position_open,
    df_position_changes,
):
    """
    Plot normalized prices together with active signal periods and trades.
    """
    import matplotlib.pyplot as plt

    signal_meta = [
        (tickers[0], "Signal 0 – SPY (VIX Term Structure)", "steelblue"),
        (tickers[1], "Signal 1 – XLE (Oil Momentum)", "darkorange"),
        (tickers[2], "Signal 2 – GLD (Gold Safe-Haven)", "goldenrod"),
    ]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 10),
        sharex=True,
        sharey=True,
    )

    fill_top = max(
        (df_prices[ticker] / df_prices[ticker].iloc[0]).max()
        for ticker, _, _ in signal_meta
    ) * 1.12

    for ax, (ticker, title, color) in zip(axes, signal_meta):
        price = df_prices[ticker]
        price_norm = price / price.iloc[0]

        ax.plot(
            price.index,
            price_norm,
            color=color,
            linewidth=1.2,
            alpha=0.90,
            label="Performance of the signal",
        )

        signal = df_position_open[ticker]
        ax.fill_between(
            signal.index,
            0,
            fill_top,
            where=signal == 1,
            alpha=0.12,
            color=color,
            step="post",
            label="Signal active",
        )

        changes = df_position_changes[ticker]
        buy_dates = changes[changes == 1].index
        sell_dates = changes[changes == -1].index
        offset = price_norm.max() * 0.06

        ax.scatter(
            buy_dates,
            price_norm.loc[buy_dates] - offset,
            marker="^",
            color="green",
            s=40,
            zorder=5,
            label="Buy",
        )
        ax.scatter(
            sell_dates,
            price_norm.loc[sell_dates] + offset,
            marker="v",
            color="red",
            s=40,
            zorder=5,
            label="Sell",
        )

        ax.set_title(
            f"{title}  |  {len(buy_dates)} Buy Times, "
            f"{len(sell_dates)} Sell Times",
            fontsize=10,
        )
        ax.set_ylabel("Growth of €1 Invested")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylim(0, fill_top)
    axes[-1].set_xlabel("Date")
    fig.suptitle("Entry Time / Exit Time", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()


def isolated_signal_curve(
    ticker,
    df_prices,
    df_price_changes,
    df_position_changes,
    initial_cash=1.0,
    capital_fraction=0.30,
):
    """
    Simulate one signal by itself using the same allocation rule as the
    combined strategy.
    """
    invested_value = 0.0
    cash_value = initial_cash
    values = []

    for index in df_prices.index:
        position_change = df_position_changes.loc[index, ticker]
        price_change = df_price_changes.loc[index, ticker]

        if position_change < 0 and invested_value > 0.0:
            cash_value += invested_value
            invested_value = 0.0

        if invested_value > 0.0:
            invested_value *= price_change

        if position_change > 0 and cash_value > 0.0:
            amount = cash_value * capital_fraction
            invested_value += amount
            cash_value -= amount

        values.append(invested_value + cash_value)

    curve = pd.Series(values, index=df_prices.index)
    return curve / curve.iloc[0]


def plot_signal_value_analysis(
    tickers,
    df_prices,
    df_price_changes,
    df_position_open,
    df_position_changes,
    df_position,
    initial_cash=1.0,
    capital_fraction=0.30,
):
    """
    Plot isolated signal curves, signal activity, and summary metrics.
    """
    import matplotlib.pyplot as plt

    signal_curves = {
        ticker: isolated_signal_curve(
            ticker,
            df_prices,
            df_price_changes,
            df_position_changes,
            initial_cash=initial_cash,
            capital_fraction=capital_fraction,
        )
        for ticker in tickers[:-1]
    }

    portfolio_values = df_position.sum(axis=1)
    portfolio_curve = portfolio_values / portfolio_values.iloc[0]
    benchmark_values = df_prices[tickers[-1]]
    benchmark_curve = benchmark_values / benchmark_values.iloc[0]
    dates = df_prices.index

    colors = {
        "SPY_VIX": "#2c7bb6",
        "XLE_OIL": "#d7191c",
        "GLD_GOLD": "#e6a817",
        "Combined": "#2ca02c",
        "Benchmark": "#777777",
    }
    labels = {
        "SPY_VIX": "Signal 0 – SPY (VIX Term-Structure)",
        "XLE_OIL": "Signal 1 – XLE (Crude Oil Momentum)",
        "GLD_GOLD": "Signal 2 – GLD (Gold Safe-Haven)",
        "Combined": "Combined Strategy",
        "Benchmark": "Benchmark AOR",
    }

    fig = plt.figure(figsize=(14, 16))
    grid = fig.add_gridspec(
        3,
        1,
        height_ratios=[3.2, 1.45, 1.35],
        hspace=0.38,
    )
    fig.suptitle("Signal Value Analysis", fontsize=15, fontweight="bold",
                 y=0.975)

    ax_perf = fig.add_subplot(grid[0])
    ax_heat = fig.add_subplot(grid[1], sharex=ax_perf)
    ax_tbl = fig.add_subplot(grid[2])

    for ticker in tickers[:-1]:
        ax_perf.plot(
            dates,
            signal_curves[ticker],
            color=colors[ticker],
            label=labels[ticker],
            linewidth=1.5,
            alpha=0.85,
        )

    ax_perf.plot(
        dates,
        portfolio_curve,
        color=colors["Combined"],
        linewidth=2.2,
        label=labels["Combined"],
    )
    ax_perf.plot(
        dates,
        benchmark_curve,
        color=colors["Benchmark"],
        linewidth=1.4,
        linestyle="--",
        alpha=0.7,
        label=labels["Benchmark"],
    )
    ax_perf.axhline(1, color="black", linewidth=0.6, linestyle=":")
    ax_perf.set_title(
        "Cumulative Value (Growth of €1 per Signal, isolated with "
        "30% allocation rule)",
        fontweight="bold",
    )
    ax_perf.set_ylabel("Portfolio Value (normalized)")
    ax_perf.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda value, _: f"€{value:.2f}")
    )
    ax_perf.legend(fontsize=8, loc="upper left")
    ax_perf.grid(alpha=0.35)
    ax_perf.spines[["top", "right"]].set_visible(False)

    for index, ticker in enumerate(tickers[:-1]):
        active = df_position_open[ticker].values.astype(float)
        ax_heat.fill_between(
            dates,
            index,
            index + active * 0.85,
            color=colors[ticker],
            alpha=0.65,
            linewidth=0,
        )

    ax_heat.set_yticks(
        [0.42, 1.42, 2.42],
        labels=[labels[ticker] for ticker in tickers[:-1]],
    )
    ax_heat.tick_params(axis="y", labelsize=9)
    ax_heat.set_ylim(0, 3)
    ax_heat.set_title("Signal Activity  (colored = invested)",
                      fontweight="bold", pad=10)
    ax_heat.grid(alpha=0.2, axis="x")
    ax_heat.spines[["top", "right", "left"]].set_visible(False)

    items = list(tickers[:-1]) + ["Combined", "Benchmark"]
    all_curves = {
        **signal_curves,
        "Combined": portfolio_curve,
        "Benchmark": benchmark_curve,
    }
    column_labels = [
        "Annualized Return",
        "Max Drawdown",
        "Sharpe Ratio",
        "Sortino Ratio",
        "Active (%)",
    ]

    data = []
    for key in items:
        curve = all_curves[key]

        if key in tickers[:-1]:
            active_values = df_position_open[key].to_numpy(dtype=float)
            active_share = np.sum(active_values) / len(active_values)
            active_text = f"{active_share * 100:.0f}%"
        else:
            active_text = "–"

        values = np.asarray(curve, dtype=float)
        returns = portfolio_returns_from_values(values)

        data.append([
            f"{annualized_return(values) * 100:+.1f}%",
            f"{drawdown_percent(values).min():.1f}%",
            f"{sharpe_ratio(returns):.2f}",
            f"{sortino_ratio(returns):.2f}",
            active_text,
        ])

    ax_tbl.axis("off")
    table = ax_tbl.table(
        cellText=data,
        rowLabels=[labels[key].split("–")[-1].strip() for key in items],
        colLabels=column_labels,
        cellLoc="center",
        loc="center",
        bbox=[0.03, 0.02, 0.94, 0.78],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)

    for index, key in enumerate(items):
        table[(index + 1, -1)].set_facecolor(colors[key] + "44")
        for column in range(len(column_labels)):
            table[(index + 1, column)].set_facecolor(colors[key] + "18")

    ax_tbl.set_title("Performance Metrics", fontweight="bold", pad=8)

    fig.subplots_adjust(top=0.94, bottom=0.04)
    plt.show()
