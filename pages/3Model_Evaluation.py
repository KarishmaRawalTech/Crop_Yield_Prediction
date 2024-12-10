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

st.markdown(" ### **🚀 Model Comparison 📊: Accuracy Analysis ✅**")
    
models = ['Linear Regression', 'Random Forest', 'XGBoost', 'SVM', 'KNN']
accuracies = [0.9006, 0.9225, 0.9214, 0.8320, 0.872]

    # Create a bar plot
plt.figure(figsize=(10, 6))
plt.bar(models, accuracies, color=['skyblue', 'lightgreen', 'salmon', 'orange', 'pink'])

    # Adding title and labels to the plot
plt.title('Model Comparison: Accuracy', fontsize=16)
plt.xlabel('Models', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)

    # Set the y-axis limit
plt.ylim([0.8, 1])

    # Display the plot in Streamlit
st.pyplot(plt)

st.write("""
             - **Interpretation**
             - **Random Forest** achieves the highest accuracy, followed by **XGBoost**, which shows a strong fit.
             - **Linear Regression** has moderate accuracy, performing better than **SVM** and **KNN** models.
             - **SVM** and **KNN** models have the lowest accuracy, suggesting they may require further tuning to improve performance.
    """)

    # Title for the Streamlit app
st.markdown(" ### **📊 Model Comparison: Cross-Validation Results & R² Scores for Regression Models 🔍**")

     
    # Define the models
models = ['Linear Regression', 'Random Forest', 'XGBoost', 'SVM', 'KNN']
Mean_accuracies = [ 0.8963, 0.9200, 0.9196, 0.8262,  0.8683]

    # Create a bar plot
plt.figure(figsize=(10, 6))
plt.bar(models, accuracies, color=['skyblue', 'lightgreen', 'salmon', 'orange', 'pink'])

    # Adding title and labels to the plot
plt.title('Model Comparison: Accuracy', fontsize=16)
plt.xlabel('Models', fontsize=12)
plt.ylabel(' Mean Accuracy', fontsize=12)

    # Set the y-axis limit
plt.ylim([0.8, 1])

    # Display the plot in Streamlit
st.pyplot(plt)

   
st.write("""
            - **Interpretation**
            - **Random Forest** shows the highest mean R² score (0.92 🎯), with a low standard deviation (0.0021 📊), indicating excellent consistency and strong model performance.
            - **XGBoost** has a similar mean R² score (0.92 🎯), with slightly higher variability (0.0027 📉), reflecting stable but slightly more fluctuating performance.
            - **Linear Regression** performs well (Mean R² = 0.90 🎯), with a small standard deviation (0.0045 📏), suggesting reliable predictions but less accuracy than the tree-based models.
            - **KNN** achieves a moderate mean R² (0.87 📈), with higher variability (0.0048 📊), indicating some instability but still reasonable performance.
            - **SVM** shows the lowest mean R² (0.83 ⚠️), with a standard deviation of 0.0039 📉, indicating relatively weak and inconsistent performance compared to the other models.

    """)
    
st.markdown("### Hyperparameter Tuning Results 🎯🔧")
st.write("""
            - **Fitting 5 folds for each of 324 candidates, totalling 1620 fits**
            - **Best Hyperparameters:**
            - `max_depth`: 30
            - `max_features`: 'sqrt'
            - `min_samples_leaf`: 1
            - `min_samples_split`: 2
            - `n_estimators`: 200
            - **Test R² (Best Model):** 0.92


         """)
st.markdown("### 🎯 Random Forest Regressor: Model Evaluation After Hyperparameter Tuning 📊")
st.write("""
            - **Interpretation**
            - **Accuracy: 91.97%** 🎯: The model predicts correctly 91.97% of the time.
            - **MSE: 3374.80** 📉: The average squared error between predicted and actual values.
            - **MAE: 46.11** 🔍: On average, the model's predictions are off by 46 units.
            - **RMSE: 58.09** 📊: The model's typical error is 58.09 units.


         """)
    
     
    
st.markdown("### **🌲 Feature Importances in Random Forest Model 🔍: Visualizing Feature Impact**")
 

# Split data into features and target
X, Y = data.drop(columns="Yield_kg_per_hectare"), data["Yield_kg_per_hectare"]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

# Train the Random Forest Regressor
rfr = RandomForestRegressor(
    max_depth=30,
    max_features="sqrt",
    min_samples_leaf=1,
    min_samples_split=2,
    n_estimators=200,
    random_state=42
)
rfr.fit(x_train, y_train)

# Predictions and evaluation metrics
pred = rfr.predict(x_test)
acc = r2_score(y_test, pred)
mse = mean_squared_error(y_test, pred)
mae = mean_absolute_error(y_test, pred)
rmse = root_mean_squared_error(y_test, pred)

# Display metrics in Streamlit
st.write(f"**Accuracy (R2 Score):** {acc:.4f}")
st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")
st.write(f"**Mean Absolute Error (MAE):** {mae:.4f}")
st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.4f}")

# Feature importance
feature_importances = rfr.feature_importances_
feature_names = x_train.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importances
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
ax.set_xlabel('Importance')
ax.set_ylabel('Features')
ax.set_title('Feature Importances in Random Forest Model')

# Display the plot in Streamlit
st.pyplot(fig)
 
st.write("""
            - **Interpretation**
             1. 🌱 **Seed_Variety** is the most important feature, having the highest impact on the model's predictions.
             2. 💧 **Irrigation_Schedule** holds significant influence on the model's output.
             3. 🌾 **Fertilizer_Amount_kg_per_hectare** contributes moderately to the model's predictions.
             4. 🌧️ **Rainfall_mm** plays a smaller, but still notable role in the model.
             5. 🦠 **Soil_Quality** has minimal importance in the model.
             6. ☀️ **Sunny_Days** is the least important feature in the Random Forest Model.


          """)

st.markdown("### **🔍SHAP Feature Importance & Insights 📊**")

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)

# Make predictions on the test set
rf_pred = rf_model.predict(x_test)

# Evaluate model performance
rf_acc = r2_score(y_test, rf_pred)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(rf_mse)

# Display model evaluation metrics in Streamlit
st.write(f"Random Forest Regression Accuracy (R2): {rf_acc}")
st.write(f"Random Forest MSE: {rf_mse}")
st.write(f"Random Forest MAE: {rf_mae}")
st.write(f"Random Forest RMSE: {rf_rmse}")

# Reduce the number of background samples for SHAP
background_data = shap.kmeans(x_train, 10)

# Create a SHAP explainer
explainer = shap.KernelExplainer(rf_model.predict, background_data)

# Get SHAP values for the test set
shap_values = explainer.shap_values(x_test)

# Plot and render SHAP summary plot in Streamlit
st.subheader("SHAP Summary Plot")
import matplotlib.pyplot as plt  # Ensure matplotlib is imported

# Render the SHAP summary plot
fig, ax = plt.subplots()
shap.summary_plot(shap_values, x_test, show=False)  # Avoid direct display
st.pyplot(fig)

# Plot feature importance using SHAP (bar plot)
st.subheader("SHAP Feature Importance (Bar Plot)")
fig, ax = plt.subplots()
shap.summary_plot(shap_values, x_test, plot_type="bar", show=False)
st.pyplot(fig)
 
st.write("""
           - **Interpretation**
- 🌊 Irrigation_Schedule: High values have a significant positive impact on the model output.
- 🌱 Seed_Variety: Different varieties have varying impacts, with some having a strong positive effect.
- 💧 Fertilizer_Amount_kg_per_hectare: Higher amounts generally increase the model output.
- 🌧️ Rainfall_mm: More rainfall tends to have a positive impact, but the effect varies.
- 🌍 Soil_Quality: Better soil quality has a moderate positive impact.
- ☀️ Sunny_Days: More sunny days have a smaller positive impact.

          """)


st.markdown("### 📊 PDP (Partial Dependence Plot) for Feature Importance 🚀")
from sklearn.inspection import PartialDependenceDisplay 
# X, Y = data.drop(columns="Yield_kg_per_hectare"), data["Yield_kg_per_hectare"]

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Choose the features for which you want to plot the PDP
features = [0, 1, 2, 3, 4, 5]  # Indices of features to plot

# Plot Partial Dependence
fig, ax = plt.subplots(figsize=(10, 8))
PartialDependenceDisplay.from_estimator(model, X_train, features=features, grid_resolution=50, ax=ax)

# Display the plot in Streamlit
st.pyplot(fig)
st.write("""
         - **Interpretation**
1. 🌱 **Soil Quality**: Better soil quality leads to a slight increase in crop yield.
2. 🌾 **Seed Variety**: Different seed varieties have a significant impact on crop yield.
3. 🌿 **Fertilizer Amount**: More fertilizer gradually increases crop yield.
4. ☀️ **Sunny Days**: More sunny days slightly improve crop yield.
5. 🌧️ **Rainfall**: Higher rainfall slightly decreases crop yield.
6. 🚿 **Irrigation Schedule**: Optimal irrigation schedules significantly boost crop yield.""")

# Save the trained model to a pickle file
with open('random_FM.pkl', 'wb') as f:
    pickle.dump(rfr, f)

print("Model saved as 'random_FM.pkl'")