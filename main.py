from src.data_loader import fetch_stock_data
from src.features import add_technical_indicators
from src.model import train_predict_model


def main():
    TICKER = "AAPL"
    START = "2020-01-01"
    END = "2024-01-01"

    try:
        raw_data = fetch_stock_data(TICKER, START, END)
        enriched_data = add_technical_indicators(raw_data)

        # New Model Step
        model, preds, actuals = train_predict_model(enriched_data)

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()