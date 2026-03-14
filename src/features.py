import pandas_ta as ta
import pandas as pd


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Adds RSI, MACD, and Bollinger Bands to the dataframe."""
    df = df.copy()

    # Simple Moving Average & RSI
    df.ta.sma(length=20, append=True)
    df.ta.rsi(length=14, append=True)

    # MACD (Moving Average Convergence Divergence)
    df.ta.macd(append=True)

    # Volatility (Bollinger Bands)
    df.ta.bbands(length=20, std=2, append=True)

    # Drop rows with NaN values created by indicators
    return df.dropna()