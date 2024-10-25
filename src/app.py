import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import datetime

import warnings
warnings.filterwarnings('ignore')

# Sample dataframe (replace this with your actual dataframe)
data = pd.read_csv("https://raw.githubusercontent.com/tushuli396/Real-estate-investment/refs/heads/main/data/Real-estate.csv")

df = data
df['X1 transaction timestamp'] = pd.to_datetime(df['X1 transaction date'], unit='s')

df.set_index('X1 transaction timestamp', inplace=True)

# Add day_of_year and time_of_day columns
df['day_of_year'] = df.index.dayofyear
df['time_of_day'] = df.index.hour + df.index.minute / 60

# Preprocess the data
features = df.drop(columns=['No', 'day_of_year', 'time_of_day', 'Y house price of unit area'])
target = df['Y house price of unit area']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train a Bagging Regressor
model = BaggingRegressor(random_state=42)
model.fit(X_train_scaled, y_train)

# Helper function to convert a date to 2012.917 format (fractional year)
def convert_to_fractional_year(date):
    year = date.year
    start_of_year = pd.Timestamp(f'{year}-01-01')
    next_year = pd.Timestamp(f'{year + 1}-01-01')
    fraction = (date - start_of_year).days / (next_year - start_of_year).days
    return year + fraction

# Streamlit UI
st.title("House Price Prediction")

# Input fields for the 6 features
st.write("Enter the following feature values:")

# Date input for transaction date
transaction_date_input = st.date_input("Enter transaction date (YYYY-MM-DD):")

# Convert the input date to fractional year format
transaction_date_fractional = convert_to_fractional_year(pd.to_datetime(transaction_date_input))

# Other features
feature_names = ['X2 house age',
                 'X3 distance to the nearest MRT station',
                 'X4 number of convenience stores', 'X5 latitude', 'X6 longitude']

# Create text inputs for the other features
input_data = [transaction_date_fractional]  # Start with the converted date
for feature in feature_names:
    value = st.text_input(f"Enter value for {feature}:", "")
    input_data.append(value)

# Predict button
if st.button("Predict"):
    # Ensure the input data is valid (convert to float)
    try:
        input_data = [float(val) for val in input_data]
    except ValueError:
        st.error("Please enter valid numeric values for all features.")
        st.stop()

    # Convert input data to a numpy array and reshape for prediction
    input_data_scaled = scaler.transform([input_data])  # Scale the input data

    # Make a prediction
    prediction = model.predict(input_data_scaled)

    # Display the result
    st.success(f"Predicted House Price: {prediction[0]:.2f}")

    # Extract latitude and longitude for the map
    latitude = input_data[4]  # X5 latitude
    longitude = input_data[5]  # X6 longitude

    # Display map showing the location
    st.write("Location of the predicted property:")
    map_data = pd.DataFrame({'lat': [latitude], 'lon': [longitude]})
    st.map(map_data)

# Add "Data Droolers" text to the bottom-right corner
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        right: 0;
        width: 100%;
        background-color: rgba(0, 0, 0, 0);
        color: grey;
        text-align: right;
        padding: 10px;
        font-size: 14px;
    }
    </style>
    <div class="footer">
        Data Droolers
    </div>
    """,
    unsafe_allow_html=True
)
