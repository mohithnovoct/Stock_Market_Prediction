import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense, Dropout
from datetime import datetime, timedelta


st.set_page_config(page_title="Stock Price Prediction", layout="wide")
st.title("📈 Stock Price Prediction - LSTM vs GRU")

# Sidebar Inputs
st.sidebar.header("Input Parameters")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL for Apple)", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=datetime(2018, 1, 1))
end_date = st.sidebar.date_input("End Date", value=datetime.now())
time_steps = st.sidebar.slider("Time Steps for LSTM/GRU", min_value=10, max_value=100, value=60)
epochs = st.sidebar.slider("Training Epochs", min_value=5, max_value=100, value=50)

# Model Selection
st.sidebar.header("Model Selection")
model_type = st.sidebar.radio("Choose Model(s)", ["LSTM Only", "GRU Only", "Compare Both"])


@st.cache_data
def fetch_stock_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return pd.DataFrame()

try:
    
    if start_date >= end_date:
        st.error("Start date must be before end date")
        st.stop()
    
    data = fetch_stock_data(ticker, start_date, end_date)
    
    if data is None or data.empty:
        st.error(f"Could not fetch data for ticker '{ticker}'. Please check if the ticker symbol is correct.")
        st.info("💡 Try using: AAPL (Apple), GOOGL (Google), MSFT (Microsoft), TSLA (Tesla)")
        st.stop()

    if len(data) < time_steps:
        st.error(f"Not enough data. Need at least {time_steps} data points, but only got {len(data)}.")
        st.stop()
    
    st.subheader(f"Stock Data for {ticker}")
    st.dataframe(data.tail())

    # Plot closing price
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
    fig.update_layout(title=f"{ticker} Closing Prices", xaxis_title="Date", yaxis_title="Price (USD)")
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------
    # Data Preparation
    # ------------------------------------------------------
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close']].values)

    X, y = [], []
    for i in range(time_steps, len(scaled_data)):
        X.append(scaled_data[i-time_steps:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Split train/test
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # ------------------------------------------------------
    # Helper Functions
    # ------------------------------------------------------
    def build_lstm_model(time_steps):
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(time_steps, 1)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def build_gru_model(time_steps):
        model = Sequential([
            GRU(units=50, return_sequences=True, input_shape=(time_steps, 1)),
            Dropout(0.2),
            GRU(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def safe_inverse_transform(scaler, arr):
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return scaler.inverse_transform(arr)

    def calculate_rmse(predictions, actual):
        return np.sqrt(np.mean((predictions.flatten() - actual.flatten())**2))

    def generate_predictions(model, X_train, X_test):
        train_predict = model.predict(X_train, verbose=0)
        test_predict = model.predict(X_test, verbose=0)
        return train_predict, test_predict

    def generate_future_predictions(model, scaled_data, time_steps, scaler, periods=30):
        last_sequence = scaled_data[-time_steps:].copy()
        future_predictions = []

        for _ in range(periods):
            input_seq = last_sequence.reshape(1, time_steps, 1)
            pred = model.predict(input_seq, verbose=0)
            future_predictions.append(pred[0, 0])
            last_sequence = np.append(last_sequence[1:], pred[0, 0])

        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
        return future_predictions

    # ------------------------------------------------------
    # Train Models Based on Selection
    # ------------------------------------------------------
    results = {}
    
    if model_type in ["LSTM Only", "Compare Both"]:
        st.subheader("🔵 Training LSTM Model...")
        with st.spinner("Training LSTM..."):
            lstm_model = build_lstm_model(time_steps)
            lstm_model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)
        st.success("✅ LSTM training complete!")
        
        lstm_train_pred, lstm_test_pred = generate_predictions(lstm_model, X_train, X_test)
        lstm_train_pred = safe_inverse_transform(scaler, lstm_train_pred)
        lstm_test_pred = safe_inverse_transform(scaler, lstm_test_pred)
        actual_prices = safe_inverse_transform(scaler, y)
        
        lstm_train_rmse = calculate_rmse(lstm_train_pred, actual_prices[:len(lstm_train_pred)])
        lstm_test_rmse = calculate_rmse(lstm_test_pred, actual_prices[len(lstm_train_pred):])
        
        results['LSTM'] = {
            'model': lstm_model,
            'train_pred': lstm_train_pred,
            'test_pred': lstm_test_pred,
            'train_rmse': lstm_train_rmse,
            'test_rmse': lstm_test_rmse
        }

    if model_type in ["GRU Only", "Compare Both"]:
        st.subheader("🟢 Training GRU Model...")
        with st.spinner("Training GRU..."):
            gru_model = build_gru_model(time_steps)
            gru_model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)
        st.success("✅ GRU training complete!")
        
        gru_train_pred, gru_test_pred = generate_predictions(gru_model, X_train, X_test)
        gru_train_pred = safe_inverse_transform(scaler, gru_train_pred)
        gru_test_pred = safe_inverse_transform(scaler, gru_test_pred)
        actual_prices = safe_inverse_transform(scaler, y)
        
        gru_train_rmse = calculate_rmse(gru_train_pred, actual_prices[:len(gru_train_pred)])
        gru_test_rmse = calculate_rmse(gru_test_pred, actual_prices[len(gru_train_pred):])
        
        results['GRU'] = {
            'model': gru_model,
            'train_pred': gru_train_pred,
            'test_pred': gru_test_pred,
            'train_rmse': gru_train_rmse,
            'test_rmse': gru_test_rmse
        }

    # ------------------------------------------------------
    # Display Performance Comparison
    # ------------------------------------------------------
    st.subheader("📊 Model Performance Comparison")
    
    if model_type == "Compare Both":
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🔵 LSTM Train RMSE", f"{results['LSTM']['train_rmse']:.2f}")
            st.metric("🔵 LSTM Test RMSE", f"{results['LSTM']['test_rmse']:.2f}")
        with col2:
            st.metric("🟢 GRU Train RMSE", f"{results['GRU']['train_rmse']:.2f}")
            st.metric("🟢 GRU Test RMSE", f"{results['GRU']['test_rmse']:.2f}")
        
        # Winner determination
        st.write("---")
        if results['LSTM']['test_rmse'] < results['GRU']['test_rmse']:
            st.success(f"🏆 **LSTM performs better** with {results['LSTM']['test_rmse']:.2f} test RMSE vs GRU's {results['GRU']['test_rmse']:.2f}")
        elif results['GRU']['test_rmse'] < results['LSTM']['test_rmse']:
            st.success(f"🏆 **GRU performs better** with {results['GRU']['test_rmse']:.2f} test RMSE vs LSTM's {results['LSTM']['test_rmse']:.2f}")
        else:
            st.info("🤝 Both models perform equally well!")
    else:
        model_name = "LSTM" if model_type == "LSTM Only" else "GRU"
        st.write(f"**{model_name} Train RMSE:** {results[model_name]['train_rmse']:.2f}")
        st.write(f"**{model_name} Test RMSE:** {results[model_name]['test_rmse']:.2f}")

    # ------------------------------------------------------
    # Plot Predictions
    # ------------------------------------------------------
    st.subheader("📈 Actual vs Predicted Prices")
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Actual Price', 
                              line=dict(color='white', width=2)))
    
    for model_name, result in results.items():
        color = 'cyan' if model_name == 'LSTM' else 'lime'
        
        # Train predictions
        train_plot = np.full(len(data), np.nan)
        train_plot[time_steps:time_steps+len(result['train_pred'])] = result['train_pred'].flatten()
        
        # Test predictions
        test_plot = np.full(len(data), np.nan)
        test_plot[time_steps+len(result['train_pred']):time_steps+len(result['train_pred'])+len(result['test_pred'])] = result['test_pred'].flatten()
        
        fig2.add_trace(go.Scatter(x=data.index, y=test_plot, name=f'{model_name} Test Predictions',
                                  line=dict(color=color, width=2)))
    
    fig2.update_layout(title=f"{ticker} - Actual vs Predicted Prices",
                       xaxis_title="Date", yaxis_title="Price (USD)",
                       hovermode='x unified')
    st.plotly_chart(fig2, use_container_width=True)

    # ------------------------------------------------------
    # Future Forecast
    # ------------------------------------------------------
    st.subheader("🔮 Next 30 Days Forecast")
    
    future_dates = pd.bdate_range(start=data.index[-1] + timedelta(days=1), periods=30)
    
    fig3 = go.Figure()
    
    historical_window = min(60, len(data))
    fig3.add_trace(go.Scatter(
        x=data.index[-historical_window:], 
        y=data['Close'][-historical_window:], 
        mode='lines',
        name='Historical Price',
        line=dict(color='white', width=2)
    ))
    
    forecast_data = []
    
    for model_name, result in results.items():
        future_predictions = generate_future_predictions(
            result['model'], scaled_data, time_steps, scaler, periods=30
        )
        
        color = 'cyan' if model_name == 'LSTM' else 'lime'
        fig3.add_trace(go.Scatter(
            x=future_dates, 
            y=future_predictions.flatten(), 
            mode='lines+markers',
            name=f'{model_name} Forecast',
            line=dict(color=color, width=2, dash='dash'),
            marker=dict(size=4)
        ))
        
        forecast_data.append({
            'model': model_name,
            'prediction': future_predictions[-1][0],
            'change': future_predictions[-1][0] - data['Close'][-1],
            'change_pct': ((future_predictions[-1][0] - data['Close'][-1]) / data['Close'][-1]) * 100
        })
    
    fig3.update_layout(
        title=f"{ticker} - Next 30 Days Forecast Comparison",
        xaxis_title="Date", 
        yaxis_title="Price (USD)",
        hovermode='x unified',
        showlegend=True
    )
    st.plotly_chart(fig3, use_container_width=True)
    
    # Forecast metrics
    if model_type == "Compare Both":
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Price", f"${data['Close'][-1]:.2f}")
        with col2:
            st.metric(f"🔵 LSTM 30-Day Forecast", 
                     f"${forecast_data[0]['prediction']:.2f}",
                     f"{forecast_data[0]['change_pct']:.2f}%")
        with col3:
            st.metric(f"🟢 GRU 30-Day Forecast", 
                     f"${forecast_data[1]['prediction']:.2f}",
                     f"{forecast_data[1]['change_pct']:.2f}%")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current Price", f"${data['Close'][-1]:.2f}")
        with col2:
            model_name = forecast_data[0]['model']
            icon = "🔵" if model_name == "LSTM" else "🟢"
            st.metric(f"{icon} {model_name} 30-Day Forecast", 
                     f"${forecast_data[0]['prediction']:.2f}",
                     f"{forecast_data[0]['change_pct']:.2f}%")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")