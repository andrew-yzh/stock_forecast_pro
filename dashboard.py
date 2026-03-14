

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta

from src.data_loader import fetch_stock_data
from src.features import add_technical_indicators
from src.model import train_predict_model
from src.backtest import simulate_trading, calculate_metrics

# --- UI Configuration ---
st.set_page_config(page_title="Quant Forecast Pro", layout="wide")
st.title("📈 Algorithmic Stock Price Predictor")
st.markdown("An end-to-end machine learning pipeline forecasting daily closing prices.")

# --- Sidebar Controls ---
st.sidebar.header("Model Parameters")
ticker = st.sidebar.text_input("Enter Ticker Symbol", value="AAPL").upper()
start_date = st.sidebar.date_input("Start Date", value=datetime.today() - timedelta(days=1500))
end_date = st.sidebar.date_input("End Date", value=datetime.today())

# --- Execution ---
if st.sidebar.button("Run Analysis"):
    with st.spinner(f"Fetching live data for {ticker} and training model..."):
        try:
            # 1. Run our pipeline
            raw_data = fetch_stock_data(ticker, start_date, end_date)
            enriched_data = add_technical_indicators(raw_data)
            model, predictions, actuals = train_predict_model(enriched_data)

            # 2. Display Metrics
            st.subheader(f"Model Evaluation: {ticker}")
            # Calculate a quick directional accuracy (did it guess the trend right?)
            direction_correct = ((predictions > actuals.shift(1)) == (actuals > actuals.shift(1))).mean() * 100

            col1, col2, col3 = st.columns(3)
            col1.metric("Data Points", len(enriched_data))
            col2.metric("Test Set Size", len(actuals))
            col3.metric("Directional Accuracy", f"{direction_correct:.2f}%")

            # 3. Plotting with Plotly
            st.subheader("Predicted vs Actual Prices (Test Set)")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=actuals.index, y=actuals.values, mode='lines', name='Actual Price',
                                     line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=actuals.index, y=predictions, mode='lines', name='Predicted Price',
                                     line=dict(color='orange', dash='dot')))

            fig.update_layout(xaxis_title="Date", yaxis_title="Price (USD)", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
            # --- NEW BACKTESTING SECTION ---
            st.markdown("---")
            st.subheader("Financial Backtest ($10,000 Initial Capital)")

            # Run the simulation
            portfolio = simulate_trading(actuals, predictions)
            total_ret, sharpe, max_dd = calculate_metrics(portfolio)

            # Display Financial Metrics
            bcol1, bcol2, bcol3 = st.columns(3)
            bcol1.metric("Strategy Total Return", f"{total_ret:.2f}%")
            bcol2.metric("Sharpe Ratio", f"{sharpe:.2f}")
            bcol3.metric("Max Drawdown", f"{max_dd:.2f}%")

            # Plot the Equity Curve
            fig_bt = go.Figure()
            fig_bt.add_trace(
                go.Scatter(x=portfolio.index, y=portfolio['Strategy_Value'], mode='lines', name='Algorithmic Strategy',
                           line=dict(color='green', width=2)))
            fig_bt.add_trace(
                go.Scatter(x=portfolio.index, y=portfolio['Buy_Hold_Value'], mode='lines', name='Buy & Hold Baseline',
                           line=dict(color='gray', dash='dash')))

            fig_bt.update_layout(xaxis_title="Date", yaxis_title="Portfolio Value (USD)", hovermode="x unified")
            st.plotly_chart(fig_bt, use_container_width=True)

            # 4. Raw Data Expander
            with st.expander("View Raw Engineered Data"):
                st.dataframe(enriched_data.tail())

        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    st.info("👈 Set your parameters in the sidebar and click 'Run Analysis'.")