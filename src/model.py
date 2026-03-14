import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd


def train_predict_model(df: pd.DataFrame):
    """Trains an XGBoost model to predict the next day's Close price."""

    # Target: We want to predict tomorrow's 'Close' price
    df['Target'] = df['Close'].shift(-1)
    df = df.dropna()

    # Features: Use all columns except the Target
    X = df.drop(columns=['Target'])
    y = df['Target']

    # Split the data chronologically
    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Initialize and train XGBoost
    model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"--- XGBoost Training Complete ---")
    print(f"Mean Absolute Error: ${mae:.2f}")
    print(f"R2 Score: {r2:.4f}")

    return model, predictions, y_test