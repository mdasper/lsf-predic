from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np

# Load the model pipeline
with open('model_pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ensure all required fields are present in the form data
        form_data = request.form.to_dict()

        # Use default values for missing features (if applicable)
        default_values = {feature: 0.0 for feature in pipeline['imputer'].feature_names_in_}
        data = {feature: float(form_data.get(feature, default_values[feature])) for feature in pipeline['imputer'].feature_names_in_}

        # Convert the data into an array
        input_data = np.array([data[feature] for feature in data]).reshape(1, -1)

        # Preprocess the input data using the saved pipeline
        imputed_data = pipeline['imputer'].transform(input_data)
        scaled_data = pipeline['scaler'].transform(imputed_data)
        pca_data = pipeline['pca'].transform(scaled_data)

        # Predict using the loaded model
        prediction = pipeline['model'].predict(pca_data)[0]

        # Determine if the prediction is good or not
        result = "good strength" if 92 <= prediction <= 99 else "good strength"

        # Render the result in the HTML page
        return render_template('index.html', prediction=prediction, result=result)

    except Exception as e:
        # Log or print the full error for debugging (optional)
        print(e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
