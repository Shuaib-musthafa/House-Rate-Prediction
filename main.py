from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
try:
    model = joblib.load('model.pkl')
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Model file not found. Please ensure 'model.pkl' exists in the project directory.")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check if model.pkl exists.'})
    
    try:
        # Get form data
        bedrooms = float(request.form['bedrooms'])
        bathrooms = float(request.form['bathrooms'])
        living_area = float(request.form['living_area'])
        condition = float(request.form['condition'])
        schools_nearby = float(request.form['schools_nearby'])
        
        # Create feature array
        features = np.array([[bedrooms, bathrooms, living_area, condition, schools_nearby]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Format the prediction
        predicted_price = round(prediction, 2)
        
        return render_template('result.html', 
                             prediction=predicted_price,
                             bedrooms=int(bedrooms),
                             bathrooms=int(bathrooms),
                             living_area=int(living_area),
                             condition=int(condition),
                             schools_nearby=int(schools_nearby))
    
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'})

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    if model is None:
        return jsonify({'error': 'Model not loaded'})
    
    try:
        data = request.json
        features = np.array([[
            data['bedrooms'],
            data['bathrooms'], 
            data['living_area'],
            data['condition'],
            data['schools_nearby']
        ]])
        
        prediction = model.predict(features)[0]
        
        return jsonify({
            'predicted_price': round(prediction, 2),
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    if not os.path.exists('static'):
        os.makedirs('static')
    
    app.run(debug=True)