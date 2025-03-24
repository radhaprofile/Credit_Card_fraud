from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained model
model = joblib.load("fraud_detection_model.pkl")



# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return "Credit Card Fraud Detection API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get data from POST request
        data = request.get_json()
        
        # Convert data into a NumPy array
        features = np.array(data["features"]).reshape(1, -1)

        # Make prediction (1 = Fraud, 0 = Not Fraud)
        prediction = model.predict(features)[0]
        
        return jsonify({"fraud_prediction": int(prediction)})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
