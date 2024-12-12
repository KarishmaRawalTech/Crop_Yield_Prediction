import streamlit as st
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings('ignore')

# Title for the application
st.markdown('## 🌿💡 Agri-Analytics: Forecasting Yields from Soil to Sun 🌞💧')

# Load and cache the data
@st.cache_data
def load_data():
    return pd.read_csv('agricultural_yield_train.csv')

data = load_data()

# Define features and target variable
X = data.drop(columns="Yield_kg_per_hectare")
Y = data[["Yield_kg_per_hectare"]]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train.values.ravel())

# Function to make predictions
def predict_soil_quality(soil_quality, seed_variety, Fertilizer_Amount_kg_per_hectare, sunny_days, Rainfall_mm, irrigation_schedule):
    # Prepare the input data as a dictionary
    input_data = {
        'Soil_Quality': soil_quality,
        'Seed_Variety': seed_variety,
        'Fertilizer_Amount_kg_per_hectare': Fertilizer_Amount_kg_per_hectare,
        'Sunny_Days': sunny_days,
        'Rainfall_mm': Rainfall_mm,
        'Irrigation_Schedule': irrigation_schedule
    }
    # Convert dictionary to a DataFrame
    input_data_df = pd.DataFrame([input_data])
    
    # Align the input DataFrame with training columns
    for col in X_train.columns:
        if col not in input_data_df:
            input_data_df[col] = 0  # Add missing columns with default value
    input_data_df = input_data_df[X_train.columns]
    
    # Make the prediction
    prediction = model.predict(input_data_df)
    return prediction[0]

# Streamlit interface
st.title("Crop Yield Prediction")
st.write("Please enter the following details to predict the crop yield:")

# User input fields with updated limits and validations
soil_quality = st.slider("Soil Quality", min_value=50.0, max_value=100.0, value=74.76)
seed_variety = st.radio("Seed Variety (0 for Non-hybrid seed, 1 for Hybrid seed)", options=[0, 1], index=0)
Fertilizer_Amount_kg_per_hectare = st.number_input("Fertilizer Amount (kg per hectare)", value=175.18)
sunny_days = st.number_input("Sunny Days", value=99.93)
Rainfall_mm = st.number_input("Rainfall (mm)", value=500.53)
irrigation_schedule = st.number_input("Irrigation Schedule", min_value=0.0, max_value=15.0, value=5.03)

# Predict and display result
if st.button('Predict'):
    prediction = predict_soil_quality(soil_quality, seed_variety, Fertilizer_Amount_kg_per_hectare, sunny_days, Rainfall_mm, irrigation_schedule)
    st.write(f"The predicted crop yield is: {prediction:.2f} kg per hectare")