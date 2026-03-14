import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
import pandas as pd

from src.data_loader import fetch_stock_data
from src.features import add_technical_indicators


def find_best_parameters():
    print("Fetching data and engineering features...")
    # Get a good chunk of data for robust tuning
    raw_data = fetch_stock_data("AAPL", "2018-01-01", "2024-01-01")
    df = add_technical_indicators(raw_data)

    # Setup Target and Features
    df['Target'] = df['Close'].shift(-1)
    df = df.dropna()

    X = df.drop(columns=['Target'])
    y = df['Target']

    # Chronological split (keep 20% untouched for the final real-world test)
    split = int(len(df) * 0.8)
    X_train, y_train = X[:split], y[:split]

    # 1. Define the grid of parameters to test
    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

    # 2. Initialize the base model
    base_model = xgb.XGBRegressor(random_state=42)

    # 3. Setup Time Series Cross-Validation (prevents data leakage)
    tscv = TimeSeriesSplit(n_splits=3)

    # 4. Configure the Randomized Search
    print("Starting hyperparameter search. This will test 20 different configurations...")
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=20,  # Number of random combinations to try
        scoring='neg_mean_absolute_error',
        cv=tscv,
        verbose=1,
        random_state=42,
        n_jobs=-1  # Uses all your CPU cores
    )

    # Run the search
    search.fit(X_train, y_train)

    print("\n" + "=" * 40)
    print("🏆 TUNING COMPLETE 🏆")
    print("=" * 40)
    print("Copy and paste these exact parameters into your src/model.py file:\n")

    best_params = search.best_params_
    for key, value in best_params.items():
        print(f"{key}={value},")


if __name__ == "__main__":
    find_best_parameters()