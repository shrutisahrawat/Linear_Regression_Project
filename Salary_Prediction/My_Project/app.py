import sys
import os
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Initialize Flask App
app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# Home route to serve HTML page
@app.route("/", methods=["GET"])
def Home():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict_salary():
    try:
        # Extract input data
        float_features = [float(x) for x in request.form.values()]
        features = np.array(float_features).reshape(1, -1)  # Reshape to 2D array

        # Predict salary
        prediction = model.predict(features)

        # Return JSON response
        return jsonify({"result": f"The predicted salary is {prediction[0]}"})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, port=5000)
