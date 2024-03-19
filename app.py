import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from datetime import datetime

def load_data(stock_symbol, start_date, end_date):
    try:
        # Fetch historical stock price data
        stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

        # Check if stock data is empty (ticker does not exist)
        if stock_data.empty:
            st.error("Stock symbol does not exist. Please enter a valid symbol.")
            return None

        # Extract features (e.g., closing price) and target variable (next day's closing price)
        stock_data['NextClose'] = stock_data['Close'].shift(-1)
        stock_data.dropna(inplace=True)

        return stock_data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def train_model(X_train, y_train, model_type):
    try:
        # Train the selected machine learning model
        if model_type == 'Linear Regression':
            model = LinearRegression()
        elif model_type == 'Decision Tree':
            model = DecisionTreeRegressor(random_state=42)
        elif model_type == 'Random Forest':
            model = RandomForestRegressor(random_state=42)
        else:
            st.error("Invalid model selection.")
            return None
        
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        st.error(f"Error training model: {e}")
        return None
    
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
    - It then extracts features such as the opening price, closing price, highest price, lowest price, and trading volume.
    - The model uses this historical data to train a machine learning model selected by the user, which learns patterns in the data to predict the next day's closing price.
    - After training, the model is evaluated using mean squared error (MSE), which measures the average squared difference between the actual and predicted values.
    - Finally, the model predicts the next day's closing price based on the most recent data available.
    """)

    st.title('Stock Price Predictor')
    st.subheader('Not financial advice!')
    
    # User inputs
    stock_symbol = st.text_input('Enter stock symbol (e.g., AAPL):')
    start_date = st.text_input('Enter start date (YYYY-MM-DD):')
    end_date = st.text_input('Enter end date (YYYY-MM-DD):')
    model_type = st.selectbox('Select machine learning model:', ['Linear Regression', 'Decision Tree', 'Random Forest'])
    
    # Load data
    if st.button('Fetch Data'):
        if not stock_symbol:
            st.error("Please enter a valid stock symbol.")
            return
        if not validate_date(start_date) or not validate_date(end_date):
            st.error("Please enter valid start and end dates in the format YYYY-MM-DD.")
            return
        stock_data = load_data(stock_symbol, start_date, end_date)
        st.write(stock_data)

        # Define features and target variable
        X = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
        y = stock_data['NextClose']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        model = train_model(X_train, y_train, model_type)

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
