import streamlit as st
import yfinance as yf
import pandas as pd
import ta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from datetime import datetime

# Function to load stock data and calculate technical indicators
def load_data(stock_symbol, start_date, end_date, selected_indicators):
    try:
        # Fetch historical stock price data
        stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

        # Check if stock data is empty (ticker does not exist)
        if stock_data.empty:
            st.error("Stock symbol does not exist. Please enter a valid symbol.")
            return None

        # Calculate selected technical indicators
        for indicator in selected_indicators:
            if indicator == 'SMA':
                stock_data['SMA'] = ta.trend.sma_indicator(stock_data['Close'], window=20)
            elif indicator == 'EMA':
                stock_data['EMA'] = ta.trend.ema_indicator(stock_data['Close'], window=20)
            # Add more technical indicators as needed

        # Extract features (e.g., closing price) and target variable (next day's closing price)
        stock_data['NextClose'] = stock_data['Close'].shift(-1)
        stock_data.dropna(inplace=True)

        return stock_data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Function to train the linear regression model
def train_model(X_train, y_train):
    try:
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        st.error(f"Error training model: {e}")
        return None

# Function to validate date format
def validate_date(date_text):
    try:
        datetime.strptime(date_text, '%Y-%m-%d')
        return True
    except ValueError:
        return False

def main():
    st.subheader("How Predictions are Made:")
    st.markdown("""
    - The application fetches historical stock price data from Yahoo Finance based on the provided stock symbol, start date, and end date.
    - It then calculates selected technical indicators such as Simple Moving Average (SMA), Exponential Moving Average (EMA), etc.
    - The model uses this historical data to train a linear regression model, which learns patterns in the data to predict the next day's closing price.
    - After training, the model is evaluated using mean squared error (MSE), which measures the average squared difference between the actual and predicted values.
    - Finally, the model predicts the next day's closing price based on the most recent data available.
    """)

    st.title('Stock Price Predictor')
    st.subheader('Not financial advice!')

    # User inputs
    stock_symbol = st.text_input('Enter stock symbol (e.g., AAPL):')
    start_date = st.text_input('Enter start date (YYYY-MM-DD):')
    end_date = st.text_input('Enter end date (YYYY-MM-DD):')
    
    # Checkbox for selecting technical indicators
    st.sidebar.subheader('Technical Indicators')
    selected_indicators = st.sidebar.multiselect(
        'Select Technical Indicators to Display:',
        ['SMA', 'EMA'],  # Add more indicators as needed
        default=['SMA']
    )

    # Load data
    if st.button('Fetch Data'):
        if not stock_symbol:
            st.error("Please enter a valid stock symbol.")
            return
        if not validate_date(start_date) or not validate_date(end_date):
            st.error("Please enter valid start and end dates in the format YYYY-MM-DD.")
            return
        stock_data = load_data(stock_symbol, start_date, end_date, selected_indicators)
        st.write(stock_data)

        if stock_data is not None:
            # Define features and target variable
            X = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
            y = stock_data['NextClose']

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train the model
            model = train_model(X_train, y_train)

            # Make predictions on the test set
            if model is not None:
                y_pred = model.predict(X_test)

                # Evaluate the model
                mse = mean_squared_error(y_test, y_pred)
                st.write(f"Mean Squared Error: {mse}")

                # Example: Predict the next day's closing price
                last_day_data = X[-1:].values  # Use the last day's data as input
                next_day_pred = model.predict(last_day_data)
                st.write(f"Predicted next day's closing price: {next_day_pred[0]}")

if __name__ == '__main__':
    main()
