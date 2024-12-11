from flask import Flask, request, jsonify
import numpy as np
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
MODEL_PATH = 'RandomForest.pkl'  # Ensure the file is in the same directory or update the path accordingly
with open(MODEL_PATH, 'rb') as file:
    RF = pickle.load(file)

# Define prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Parse input JSON
    data = request.json
    try:
        # Convert input data to a NumPy array
        input_features = np.array([[
            data['N'], data['P'], data['K'], data['temperature'],
            data['humidity'], data['ph'], data['rainfall']
        ]])
        # Make prediction
        prediction = RF.predict(input_features)
        return jsonify({'prediction': prediction[0]})
    except KeyError as e:
        return jsonify({'error': f'Missing key: {e}'}), 400

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
