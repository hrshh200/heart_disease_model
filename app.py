# app.py
import streamlit as st
import pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# Load the model
with open('./heart_disease_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Create a prediction endpoint
@app.route('/', methods=['POST'])
def predict():
    data = request.get_json()  # Get JSON data from the request
    input_data = np.array([[data['age'], data['sex'], data['cp'], data['trest'], data['chol'],
                            data['fbs'], data['rest'], data['thala'], data['exang'],
                            data['oldpeak'], data['slope'], data['ca'], data['thal']]])

    # Make a prediction
    prediction = model.predict(input_data)
    result = "This person is having a heart disease" if prediction[0] == 1 else "This person is not having a heart disease"
    return jsonify({"prediction": result})

# Run the Streamlit interface as usual
def main():
    st.title("Heart Disease Prediction App")
    st.write("Use the API endpoint /predict to get predictions.")

if __name__ == '__main__':
    main()
