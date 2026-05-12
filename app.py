from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    study_hours = float(request.form["study_hours"])
    attendance = float(request.form["attendance"])
    sleep_hours = float(request.form["sleep_hours"])
    previous_marks = float(request.form["previous_marks"])

    features = np.array([[study_hours, attendance, sleep_hours, previous_marks]])

    prediction = model.predict(features)[0]

    return render_template(
        "index.html",
        prediction_text=f"Predicted Final Marks: {prediction:.2f}"
    )

if __name__ == "__main__":
    app.run(debug=True)