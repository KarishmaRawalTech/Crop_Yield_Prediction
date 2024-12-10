import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error
import shap
import pickle
import joblib
from scipy.stats import shapiro
st.markdown('## 🌿💡 Agri-Analytics: Forecasting Yields from Soil to Sun 🌞💧')
@st.cache_data
def load_data():
    # Replace 'your_dataset.csv' with the actual file name
    return pd.read_csv('After_Eda.csv')

data = load_data()
st.markdown(" ## Hypothesis Testing: 🧠📊🔍")

    # Hypothesis Testing - Shapiro-Wilk Normality Test 🧪📊
st.subheader("Shapiro-Wilk Normality Test 🧪📊")

X = data.drop(columns="Yield_kg_per_hectare")
Y = data[["Yield_kg_per_hectare"]]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Check if the number of features in X_train and X_test are aligned
print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

    # Random Forest Model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

    # Model Prediction
y_pred = rf.predict(X_test)  # Note: Use rf instead of rfr, as that was likely a typo

    # Ensure y_test is a 1D series and y_pred is also 1D
y_test = y_test.values.flatten()
y_pred = y_pred.flatten()

    # Calculate residuals
residuals = y_test - y_pred

    # Perform Shapiro-Wilk test for normality of residuals
stat, p_value = shapiro(residuals)
print(f"Shapiro-Wilk Statistic: {stat}, P-value: {p_value}")

    # Interpretation of the p-value
if p_value < 0.05:
    print("The residuals are not normally distributed (reject H0).")
else:
    print("The residuals are normally distributed (fail to reject H0).")

    # Plotting the Residuals
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
plt.title('Residuals Histogram')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
st.pyplot(plt)

    # Interpretation
st.subheader("Interpretation")
st.write("""
    - The Shapiro-Wilk test generates a test statistic close to 1 if the data are normally distributed.
    - Normally distributed residuals indicate that the model's errors are random and the model is unbiased.
    """)

st.subheader("Permutation Importance 📊🔍")
    
    # Permutation Importance
from sklearn.inspection import permutation_importance
result = permutation_importance(rf, X, Y, n_repeats=10, random_state=42)
    
    # Plotting the Permutation Importance
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(X.columns, result.importances_mean, color='skyblue')
ax.set_xlabel('Permutation Importance')
ax.set_title('Permutation Importance of Features')
st.pyplot(fig)
    
    # Interpretation of Permutation Importance 📊🔍
st.write("""
    **Interpretation of Permutation Importance:**
    
    Permutation Importance measures the decrease in model performance (e.g., R², accuracy) when the values of a particular feature are randomly shuffled, disrupting its relationship with the target variable. 
    
    - **🌾 Seed_Variety**: This feature has the highest importance, indicating it is the most critical predictor of yield. Changes to this feature have the greatest impact on model performance.
    - **💧 Irrigation_Schedule**: The second most important feature, suggesting that the irrigation plan significantly influences the target.
    - **☀️ Sunny_Days** and **🌱 Soil_Quality**: These have the least importance, implying they have a minimal effect on the model's predictions.
    """)