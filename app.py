# app.py
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("car_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # User inputs
        year = int(request.form["year"])
        present_price = float(request.form["present_price"])
        kms_driven = int(request.form["kms_driven"])
        owner = int(request.form["owner"])
        fuel_type = int(request.form["fuel_type"])       # 0=Petrol,1=Diesel,2=CNG
        seller_type = int(request.form["seller_type"])   # 0=Dealer,1=Individual
        transmission = int(request.form["transmission"]) # 0=Manual,1=Automatic

        # Age calculation
        age = 2025 - year

        # Fuel Type One-Hot Encoding
        if fuel_type == 0:   # Petrol
            fuel = [1,0]
        elif fuel_type == 1: # Diesel
            fuel = [0,1]
        else:                # CNG
            fuel = [0,0]

        # Seller Type
        seller = [seller_type]

        # Transmission
        trans = [transmission]

        # Combine all features (total 8 features)
        features = np.array([[age, present_price, kms_driven, owner] + fuel + seller + trans])

        # Make prediction
        prediction = model.predict(features)[0]

        return render_template("index.html", prediction_text=f"Predicted Price: ₹ {prediction:.2f} Lakh")

if __name__ == "__main__":
    app.run(debug=True)
