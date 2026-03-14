import pandas as pd
import numpy as np


def simulate_trading(actuals: pd.Series, predictions: np.ndarray, initial_capital: float = 10000.0) -> pd.DataFrame:
    """Simulates a trading strategy based on model predictions."""
    # Align actuals and predictions
    df = pd.DataFrame({'Actual': actuals, 'Predicted': predictions})

    # We compare tomorrow's predicted price with today's actual close
    df['Actual_Today'] = df['Actual'].shift(1)
    df = df.dropna()

    # Trading Signal: 1 (Buy/Hold) if we predict the price will go up, 0 (Cash) if down
    df['Signal'] = np.where(df['Predicted'] > df['Actual_Today'], 1, 0)

    # Calculate daily percent returns of the underlying stock
    df['Stock_Return'] = df['Actual'].pct_change()

    # Strategy return: We only capture the stock's return if our signal from yesterday was 1
    df['Strategy_Return'] = df['Signal'].shift(1) * df['Stock_Return']
    df = df.dropna()

    # Calculate cumulative portfolio growth
    df['Strategy_Value'] = initial_capital * (1 + df['Strategy_Return']).cumprod()
    df['Buy_Hold_Value'] = initial_capital * (1 + df['Stock_Return']).cumprod()

    return df


def calculate_metrics(portfolio_df: pd.DataFrame):
    """Calculates Total Return, Sharpe Ratio, and Max Drawdown."""
    # Total Return
    total_return = (portfolio_df['Strategy_Value'].iloc[-1] / portfolio_df['Strategy_Value'].iloc[0]) - 1

    # Annualized Sharpe Ratio (assuming roughly 4% risk-free rate)
    rf_daily = 0.04 / 252
    excess_returns = portfolio_df['Strategy_Return'] - rf_daily

    if excess_returns.std() != 0:
        sharpe_ratio = np.sqrt(252) * (excess_returns.mean() / excess_returns.std())
    else:
        sharpe_ratio = 0.0

    # Maximum Drawdown (biggest drop from a peak)
    rolling_max = portfolio_df['Strategy_Value'].cummax()
    drawdown = (portfolio_df['Strategy_Value'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    return total_return * 100, sharpe_ratio, max_drawdown * 100