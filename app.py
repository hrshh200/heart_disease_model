# app.py
import streamlit as st
import pickle
import numpy as np

# Load the model
with open('./model/heart_disease_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Title and description
st.title("Heart Disease Prediction App")
st.write("Enter the fields to get the results: ")

# Input fields for user data
age = st.slider("age", 1, 100, 50)
sex = st.slider("sex ", 0, 1, 0)
cp = st.slider("cp", 1, 10, 2)
trestbps = st.slider("trestbps", 100, 200, 120)
chol = st.slider("chol", 100, 300, 200)
fbs = st.slider("fbs", 0, 1, 0)
restecg = st.slider("restecg", 0,1,0)
thalach = st.slider("thalach", 100, 200, 150)
exang = st.slider("exang", 0, 1, 0)
oldpeak = st.slider("oldpeak", 0.0, 7.0, 1.3)
slope = st.slider("slope", 0, 3, 2)
ca = st.slider("ca", 0, 4 , 2)
thal = st.slider("thal", 0 , 5, 2)

# Prediction button
if st.button("Predict"):
    # Create a NumPy array for model input
    input_data = np.array([[age, sex, cp, trestbps,chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

    # Make a prediction
    prediction = model.predict(input_data)
    if(prediction[0] == 1):
        st.write("The person is having a heart disease")
    else:
        st.write("The person is not having a heart disease")
