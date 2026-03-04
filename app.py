import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')

# Original pixel columns
original_px_columns = [f'px_{i}' for i in range(768)]

# Drop px_26 (same as training)
feature_names = [col for col in original_px_columns if col != 'px_26']

# Class mapping
class_names = {
    0: 'human',
    1: 'elephant',
    2: 'empty'
}

# -------------------------------
# Root endpoint (Fixes 404 error)
# -------------------------------
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "running",
        "service": "Elephant Detection API",
        "endpoint": "/predict",
        "expected_pixels": 768
    })

# -------------------------------
# Health check (useful for Render)
# -------------------------------
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

# -------------------------------
# Prediction endpoint
# -------------------------------
@app.route('/predict', methods=['POST'])
def predict():

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    if 'pixel_values' not in data:
        return jsonify({"error": "Missing 'pixel_values' in JSON payload"}), 400

    raw_pixel_values = data['pixel_values']

    if len(raw_pixel_values) != 768:
        return jsonify({
            "error": f"Expected 768 pixel values but got {len(raw_pixel_values)}"
        }), 400

    processed_pixel_values = []

    for i, val in enumerate(raw_pixel_values):

        # Skip px_26
        if i == 26:
            continue

        try:
            processed_pixel_values.append(float(val))
        except:
            processed_pixel_values.append(np.nan)

    # Convert to dataframe
    input_df = pd.DataFrame([processed_pixel_values], columns=feature_names)

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction_numeric = model.predict(input_scaled)[0]
    predicted_class = class_names[prediction_numeric]

    # Probability
    probability = float(np.max(model.predict_proba(input_scaled)))

    return jsonify({
        "prediction": predicted_class,
        "confidence": round(probability, 3)
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
