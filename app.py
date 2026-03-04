import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define the feature names, excluding 'px_26' which was dropped during training
# Assuming X_train.columns from the notebook contains the correct feature names
# In a real scenario, you would save X_train.columns as a separate file or embed in the model for consistency
# For this example, we'll recreate the feature names based on the original preprocessing steps.
# The original X had 768 columns (px_0 to px_767). px_26 was dropped.
original_px_columns = [f'px_{i}' for i in range(768)]
feature_names = [col for col in original_px_columns if col != 'px_26']

# Map numerical predictions to class names
class_names = {0: 'human', 1: 'elephant', 2: 'empty'}

@app.route('/predict', methods=['POST'])
def predict():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    if 'pixel_values' not in data:
        return jsonify({"error": "Missing 'pixel_values' in JSON payload"}), 400

    raw_pixel_values = data['pixel_values']

    if len(raw_pixel_values) != len(feature_names) + 1: # +1 for the dropped 'px_26'
        # This check is for the raw input that *would* contain 'px_26' if it were not NaN
        return jsonify({"error": f"Expected {len(feature_names) + 1} pixel values, but got {len(raw_pixel_values)}. Ensure px_26 is present if it's not handled as NaN in raw input."}), 400

    # Convert raw pixel values to a list, handling potential 'nan' strings
    processed_pixel_values = []
    for i, val in enumerate(raw_pixel_values):
        if i == 26: # px_26 index
            # Skip this feature as it was dropped during training
            continue
        if isinstance(val, str) and val.lower() == 'nan':
            processed_pixel_values.append(np.nan) # np.nan will be handled by the scaler if it was imputed
        else:
            try:
                processed_pixel_values.append(float(val))
            except ValueError:
                return jsonify({"error": f"Invalid pixel value at index {i}: {val}"}), 400

    # Convert to DataFrame with correct feature names
    # Ensure the order of features matches the training data after dropping 'px_26'
    input_df = pd.DataFrame([processed_pixel_values], columns=feature_names)

    # Apply scaling
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction_numeric = model.predict(input_scaled)[0]
    predicted_class = class_names[prediction_numeric]

    return jsonify({"prediction": predicted_class}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
