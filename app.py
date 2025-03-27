# app.py
from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load trained model
try:
    model = joblib.load('co2_emission_model.pkl')
except FileNotFoundError:
    model = None
    print("Error: Model file 'co2_emission_model.pkl' not found.")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Ensure model is loaded
    if model is None:
        return render_template('result.html', prediction="Model not found. Please check the server.")

    # Get data from form
    try:
        features = [float(request.form.get(f'feature{i}', 0)) for i in range(1, 5)]
    except ValueError:
        return render_template('result.html', prediction="Invalid input. Please enter numeric values.")

    # Make prediction
    prediction = model.predict([features])[0]

    return render_template('result.html', prediction=f"Predicted CO₂ Emission: {prediction:.2f} g/km")

if __name__ == '__main__':
    app.run(debug=True)
