# app.py
import streamlit as st
import pickle
import numpy as np

# Load the model
with open('./heart_disease_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Title and description
st.title("Heart Disease Prediction App")

params = st.experimental_get_query_params()
# Input fields for user data
if 'age' in params and 'sex' in params and 'cp' in params and 'trest' in params and 'chol' in params and 'fbs' in params and 'rest' in params and 'thala' in params and 'exang' in params and 'oldpeak' in params and 'slope' in params and 'ca' in params and 'thal' in params:
    # Convert parameters to floats
    age = float(params['age'][0])
    sex = float(params['sex'][0])
    cp = float(params['cp'][0])
    trest = float(params['trest'][0])
    chol = float(params['chol'][0])
    fbs = float(params['fbs'][0])
    rest = float(params['rest'][0])
    thala = float(params['thala'][0])
    exang = float(params['exang'][0])
    oldpeak = float(params['oldpeak'][0])
    slope = float(params['slope'][0])
    ca = float(params['ca'][0])
    thal = float(params['thal'][0])


    # Create a NumPy array for model input
    input_data = np.array([[age, sex, cp, trest ,chol, fbs, rest, thala, exang, oldpeak, slope, ca, thal]])
    # Make a prediction
    prediction = model.predict(input_data)
    if(prediction[0] == 1):
        st.write("This person is having a heart disease")
    else:
        st.write("This person is not having a heart disease")
