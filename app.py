from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os # Import os for path handling

# Initialize Flask application
app = Flask(__name__, template_folder='templates') # Specify the templates folder

# Define the path to the model file
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'random_forest_liquidity_model.pkl')

# Load the trained model
try:
    model_pipeline = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}. Make sure you run the notebook to save the model.")
    model_pipeline = None # Handle the case where the model file is missing

# Define the expected features based on your training data
# This list must match the order and names of features used during training
EXPECTED_FEATURES = [
    'price', '24h_volume', 'price_lag_1', 'price_lag_7',
    'price_rolling_mean_7', 'price_rolling_std_7',
    'year', 'month', 'day', 'dayofweek', 'dayofyear',
    'price_diff_rolling_mean_7'
]

@app.route('/')
def index():
    """
    Renders the main prediction page.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives data via POST request, makes a liquidity prediction
    using the loaded RandomForestRegressor model, and returns the prediction.
    """
    if model_pipeline is None:
        return jsonify({'error': 'Model not loaded.'}), 500

    try:
        # Get data from the POST request
        data = request.get_json(force=True) # force=True allows parsing even if content-type is not application/json

        # Convert the incoming data into a Pandas DataFrame
        # Assumes 'data' is a list of dictionaries, where each dictionary represents one data point
        # Example: [{"feature1": value1, "feature2": value2}, {"feature1": value3, "feature2": value4}]

        if not isinstance(data, list):
             # If it's a single dictionary, wrap it in a list
            data = [data]

        input_df = pd.DataFrame(data)

        # Ensure the input DataFrame has the same columns as expected.
        # This is a crucial validation step.
        if not all(feature in input_df.columns for feature in EXPECTED_FEATURES):
             missing_features = [feature for feature in EXPECTED_FEATURES if feature not in input_df.columns]
             return jsonify({'error': f'Missing required features: {missing_features}'}), 400

        # Reorder columns to match the order the model was trained on
        input_df = input_df[EXPECTED_FEATURES]


        # Make prediction using the loaded pipeline
        prediction = model_pipeline.predict(input_df)

        # Convert prediction (numpy array) to a list for JSON serialization
        prediction_list = prediction.tolist()

        # Return the prediction as JSON response
        return jsonify({'prediction': prediction_list})

    except Exception as e:
        # Log the error for debugging
        app.logger.error(f"An error occurred during prediction: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500 # Return an error message

if __name__ == '__main__':
    # Run the Flask app
    # In a production environment, you would use a more robust server like Gunicorn or uWSGI
    # Setting debug=True is useful during development
    app.run(debug=True)