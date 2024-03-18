import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def load_data(stock_symbol, start_date, end_date):
    # Fetch historical stock price data
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

    # Extract features and target variable
    stock_data['NextClose'] = stock_data['Close'].shift(-1)
    stock_data.dropna(inplace=True)

    return stock_data

def train_model(X_train, y_train):
    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def main():
    st.title('Stock Price Predictor')
    st.subheader('Not financial advice!')
    
    # User inputs
    stock_symbol = st.text_input('Enter stock symbol (e.g., AAPL):')
    start_date = st.text_input('Enter start date (YYYY-MM-DD):')
    end_date = st.text_input('Enter end date (YYYY-MM-DD):')
    
    # Load data
    if st.button('Fetch Data'):
        stock_data = load_data(stock_symbol, start_date, end_date)
        st.write(stock_data)

        # Define features and target variable
        X = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
        y = stock_data['NextClose']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        model = train_model(X_train, y_train)

        # Make predictions on the test set
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
