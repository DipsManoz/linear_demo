import pandas as pd
import numpy as np
import streamlit as st
import joblib

# Load the model
model = joblib.load('linear.pkl')

# Title of the Streamlit app
st.title("Linear Regression Model")

# Collecting input from the user
# For example, let's assume your model expects two input features 'feature1' and 'feature2'
feature1 = st.number_input("Enter total bill", value=0.0)


# Predict button
if st.button("Predict"):
    # Prepare input data for prediction
    input_data = np.array([[feature1]])
    
    # Make prediction using the loaded model
    prediction = model.predict(input_data)
    
    # Display the prediction result
    st.success(f"The predicted value is: {prediction[0]}")

# Footer of the app
st.write("This app uses a pre-trained linear regression model to make predictions.")
