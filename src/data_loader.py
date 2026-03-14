import yfinance as yf
import pandas as pd


def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetches historical stock data and flattens MultiIndex headers."""
    print(f"Fetching data for {ticker}...")

    # Download the data
    data = yf.download(ticker, start=start_date, end=end_date)

    if data.empty:
        raise ValueError(f"No data found for {ticker}. Check the symbol or dates.")

    # FIX: Flatten the MultiIndex headers created by newer yfinance versions
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Ensure the index is a DatetimeIndex (sometimes helps with plotting)
    data.index = pd.to_datetime(data.index)

    return data