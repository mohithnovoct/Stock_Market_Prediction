# Stock Price Prediction with LSTM and Streamlit

## Overview

This project implements a Long Short-Term Memory (LSTM) neural network to predict stock prices using historical data from the Yahoo Finance API (`yfinance`). The predictions and stock data are visualized through an interactive web application built with Streamlit. Users can input a stock ticker, select a date range, and configure model parameters to view historical data, model performance, and future price predictions.

**Disclaimer**: This project is for educational purposes only. Stock price predictions are not guaranteed to be accurate and should not be used for actual investment decisions.

## Features

- **Data Fetching**: Retrieves historical stock data (Open, High, Low, Close, Volume) using the Yahoo Finance API.
- **LSTM Model**: Trains an LSTM model to predict future closing prices based on historical data.
- **Interactive Web App**: Built with Streamlit, allowing users to:
  - Input a stock ticker (e.g., AAPL for Apple).
  - Select a date range for historical data.
  - Configure LSTM parameters (time steps and training epochs).
  - View raw stock data, basic statistics, and interactive visualizations.
- **Visualizations**:
  - Historical closing prices.
  - LSTM predictions (training and testing) compared to actual prices.
  - Future price predictions for the next 30 days.
- **Model Performance**: Displays Root Mean Squared Error (RMSE) for training and testing datasets.
- **Error Handling**: Gracefully handles invalid tickers or insufficient data with user-friendly error messages.

## Requirements

To run this project, you need Python 3.7+ and the following libraries:

- `streamlit`
- `yfinance`
- `pandas`
- `numpy`
- `plotly`
- `scikit-learn`
- `tensorflow`

## Installation

1. **Clone the Repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   Create a `requirements.txt` file with the following content:
   ```
   streamlit
   yfinance
   pandas
   numpy
   plotly
   scikit-learn
   tensorflow
   ```
   Install the dependencies using:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Save the Main Script**:
   Ensure the main Python script (`stock_prediction_app.py`) is in your project directory. The script is provided in the project repository or can be created based on the code provided.

2. **Run the Streamlit App**:
   Navigate to the project directory and run:
   ```bash
   streamlit run stock_prediction_app.py
   ```

3. **Access the Web App**:
   Streamlit will provide a local URL (e.g., `http://localhost:8501`). Open this URL in your web browser to interact with the app.

4. **Interact with the App**:
   - In the sidebar, enter a stock ticker (e.g., `AAPL` for Apple).
   - Select a start date (e.g., `2018-01-01`) and end date (default is today).
   - Adjust the `Time Steps for LSTM` (e.g., 60) and `Training Epochs` (e.g., 50).
   - View the stock data, statistics, historical price plot, LSTM predictions, and 30-day future predictions.

## Example

- **Ticker**: `AAPL`
- **Start Date**: `2018-01-01`
- **End Date**: `2025-05-23`
- **Time Steps**: 60
- **Epochs**: 50

The app will display:
- A table of historical stock data.
- Statistical summary (mean, std, min, max, etc.).
- A plot of historical closing prices.
- Training and testing predictions compared to actual prices.
- A 30-day future price prediction plot.
- RMSE metrics for model performance.

## Project Structure

```
stock_prediction_project/
├── stock_prediction_app.py  # Main Streamlit app and LSTM model script
├── requirements.txt         # Dependencies for the project
├── README.md                # This file
```

## Notes and Limitations

- **Data Dependency**: The app relies on the Yahoo Finance API, which may have rate limits or data availability issues. Ensure the ticker is valid and the date range contains sufficient data (more than `time_steps` days).
- **Model Performance**: LSTM models may not capture all market dynamics. Performance depends on the quality of data and hyperparameters (e.g., `time_steps`, `epochs`). Experimentation may be needed for better results.
- **Deployment**: To deploy the app, use Streamlit Community Cloud by linking the project to a GitHub repository. Follow Streamlit’s deployment guide for instructions.
- **Error Handling**: The app includes error handling for invalid tickers or insufficient data. If errors persist, check the console for detailed messages or ensure dependencies are correctly installed.

## Troubleshooting

- **Shape Mismatch Errors**: If you encounter errors like `could not broadcast input array`, ensure your `numpy`, `pandas`, and `tensorflow` versions are up-to-date, as older versions may handle array shapes differently.
- **Insufficient Data**: If the app reports "No data available," verify the ticker and date range. Use a well-known ticker (e.g., `AAPL`, `GOOGL`) and a date range with sufficient trading days.
- **Performance Issues**: Training the LSTM model can be slow for large datasets or many epochs. Reduce `epochs` or use a smaller date range for faster testing.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue on the project repository to suggest improvements or report bugs.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details (if included in the repository).

## Contact

For questions or feedback, please contact the project maintainer or open an issue on the repository.