import streamlit as st  # Import Streamlit
# Set page configuration
st.set_page_config(page_title="Crop Yield Prediction", layout="wide")
st.sidebar.selectbox('Select Option', ['Home', 'EDA', 'Hypothesis Testing', 'Model Evaluation','Model'])

# Title for the app
st.markdown('## 🌿💡 Agri-Analytics: Forecasting Yields from Soil to Sun 🌞💧')

st.markdown("""
    ### Problem Statement  
    The problem at hand is predicting the *crop yield (kg per hectare)* based on various factors, such as:  
    - 🌱 *Soil quality*  
    - 🌾 *Seed variety*  
    - 🧪 *Fertilizer usage*  
    - 🌞 *Weather conditions* (sunny days and rainfall)  
    - 💧 *Irrigation practices*  

    The goal is to understand how each feature influences the crop yield and make accurate predictions based on these features.

    ### Objective of the Model  

    - 🎯 The primary objective is to predict the *crop yield (kg per hectare)* based on input features.  
    - 🔧 By analyzing the impact of factors like soil quality, fertilizer usage, and irrigation schedules, the model aims to help *optimize farming practices* for better yields.  
    - 🧠 Provide insights to farmers and agricultural planners on how to adjust factors such as fertilizer amounts or irrigation schedules based on *expected weather conditions and soil quality*.  
    - 📊 Help farmers plan their resources effectively by predicting yields under varying conditions.
    """)