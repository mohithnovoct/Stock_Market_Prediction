import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


st.title("Stock Price Prediction with LSTM")


st.sidebar.header("Input Parameters")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL for Apple)", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=datetime(2018, 1, 1))
end_date = st.sidebar.date_input("End Date", value=datetime.now())
time_steps = st.sidebar.slider("Time Steps for LSTM", min_value=10, max_value=100, value=60)
epochs = st.sidebar.slider("Training Epochs", min_value=10, max_value=100, value=50)



@st.cache_data
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data



def prepare_data(data, time_steps):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close']].values)
    X, y = [], []
    for i in range(time_steps, len(scaled_data)):
        X.append(scaled_data[i - time_steps:i, 0])
        y.append(scaled_data[i, 0])
    return np.array(X), np.array(y), scaler



def create_lstm_model(time_steps):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(time_steps, 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model



try:
    stock_data = fetch_stock_data(ticker, start_date, end_date)

    if not stock_data.empty:
        st.subheader(f"Stock Data for {ticker}")
        st.dataframe(stock_data)
        st.subheader("Stock Data Statistics")
        st.write(stock_data.describe())

        st.subheader("Historical Closing Prices")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close Price'))
        fig.update_layout(title=f"{ticker} Closing Prices", xaxis_title="Date", yaxis_title="Price (USD)")
        st.plotly_chart(fig)

        X, y, scaler = prepare_data(stock_data, time_steps)

        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        model = create_lstm_model(time_steps)
        with st.spinner("Training LSTM model..."):
            model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)

        train_predict = model.predict(X_train, verbose=0)
        test_predict = model.predict(X_test, verbose=0)

        train_predict = scaler.inverse_transform(train_predict)
        y_train_inv = scaler.inverse_transform([y_train])
        test_predict = scaler.inverse_transform(test_predict)
        y_test_inv = scaler.inverse_transform([y_test])


        train_rmse = np.sqrt(np.mean((train_predict - y_train_inv.T) ** 2))
        test_rmse = np.sqrt(np.mean((test_predict - y_test_inv.T) ** 2))


        st.subheader("Model Performance")
        st.write(f"Train RMSE: {train_rmse:.2f}")
        st.write(f"Test RMSE: {test_rmse:.2f}")


        st.subheader("LSTM Predictions vs Actual Prices")
        train_plot = np.empty_like(stock_data['Close'])
        train_plot[:] = np.nan
        train_plot[time_steps:train_size + time_steps] = train_predict[:, 0]

        test_plot = np.empty_like(stock_data['Close'])
        test_plot[:] = np.nan
        test_plot[train_size + time_steps:len(stock_data)] = test_predict[:, 0]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Actual Price'))
        fig.add_trace(go.Scatter(x=stock_data.index, y=train_plot, mode='lines', name='Train Prediction'))
        fig.add_trace(go.Scatter(x=stock_data.index, y=test_plot, mode='lines', name='Test Prediction'))
        fig.update_layout(title=f"{ticker} Price Predictions", xaxis_title="Date", yaxis_title="Price (USD)")
        st.plotly_chart(fig)

        st.subheader("Future Price Prediction (Next 30 Days)")
        last_sequence = stock_data['Close'][-time_steps:].values.reshape(-1, 1)
        last_sequence_scaled = scaler.transform(last_sequence)
        future_predictions = []

        for _ in range(30):
            last_sequence_reshaped = last_sequence_scaled.reshape((1, time_steps, 1))
            next_pred = model.predict(last_sequence_reshaped, verbose=0)
            future_predictions.append(next_pred[0, 0])
            last_sequence_scaled = np.roll(last_sequence_scaled, -1)
            last_sequence_scaled[-1] = next_pred

        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))


        last_date = stock_data.index[-1]
        future_dates = [last_date + timedelta(days=x) for x in range(1, 31)]


        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=stock_data.index[-100:], y=stock_data['Close'][-100:], mode='lines', name='Historical Price'))
        fig.add_trace(go.Scatter(x=future_dates, y=future_predictions[:, 0], mode='lines', name='Future Prediction'))
        fig.update_layout(title=f"{ticker} Future Price Prediction", xaxis_title="Date", yaxis_title="Price (USD)")
        st.plotly_chart(fig)

    else:
        st.error("No data available for the selected ticker or date range.")
except Exception as e:
    st.error(f"An error occurred: {str(e)}")