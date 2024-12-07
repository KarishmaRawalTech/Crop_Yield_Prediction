import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error  
st.markdown('## 🌿💡 Agri-Analytics: Forecasting Yields from Soil to Sun 🌞💧')
@st.cache_data
def load_data():
    return pd.read_csv('After_Eda.csv')
data = load_data()
X = data.drop(columns="Yield_kg_per_hectare")
Y = data[["Yield_kg_per_hectare"]]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Function to make predictions
def predict_soil_quality(soil_quality, seed_variety, fertilizer_amount, sunny_days, rainfall, irrigation_schedule):
    input_data = np.array([[soil_quality, seed_variety, fertilizer_amount, sunny_days, rainfall, irrigation_schedule]])
    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit interface
st.title("Crop Yield Prediction")

st.write("Please enter the following details to predict the crop yield:")

# User input fields
soil_quality = st.number_input("Soil Quality", min_value=50.0, max_value=100.0, value=74.76)
seed_variety = st.number_input("Seed Variety", min_value=0.0, max_value=1.0, value=0.7)
fertilizer_amount = st.number_input("Fertilizer Amount (kg per hectare)", min_value=50.0, max_value=300.0, value=175.18)
sunny_days = st.number_input("Sunny Days", min_value=51.48, max_value=142.42, value=99.93)
rainfall = st.number_input("Rainfall (mm)", min_value=110.0, max_value=872.35, value=500.53)
irrigation_schedule = st.number_input("Irrigation Schedule", min_value=0.0, max_value=15.0, value=5.03)

# Predict and display result
if st.button('Predict'):
    prediction = predict_soil_quality(soil_quality, seed_variety, fertilizer_amount, sunny_days, rainfall, irrigation_schedule)
    st.write(f"The predicted crop yield is: {prediction:.2f} kg per hectare")

 