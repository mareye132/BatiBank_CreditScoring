from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model_path = 'C:/Users/user/Desktop/Github/BatiBank_CreditScoring/scripts/Tuned_Random_Forest.pkl'
pipeline = joblib.load(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON input data
        data = request.json
        
        # Log the raw input data
        print("Raw Input Data:", data)
        
        # Prepare input DataFrame
        input_data = data['input']
        input_df = pd.DataFrame(input_data)

        # Log the final DataFrame structure
        print("Final DataFrame for Prediction:", input_df)

        # Make prediction using the pipeline
        prediction = pipeline.predict(input_df)

        return jsonify({'prediction': prediction.tolist()})
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
