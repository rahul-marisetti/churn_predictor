from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pickle

# Load the model
with open("model1.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/analysis', methods=['GET'])
def analysis():
    return render_template("analysis.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        feature_names = [
            "CreditScore", "Geography", "Gender", "Age", "Tenure", 
            "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", 
            "EstimatedSalary"
        ]

        input_features = [
            float(request.form["CreditScore"]),
            int(request.form["Geography"]),
            int(request.form["Gender"]),
            int(request.form["Age"]),
            int(request.form["Tenure"]),
            float(request.form["Balance"]),
            int(request.form["NumOfProducts"]),
            int(request.form["HasCrCard"]),
            int(request.form["IsActiveMember"]),
            float(request.form["EstimatedSalary"])
        ]

        prediction = model.predict([input_features])[0]

        # Try to extract feature importance or coefficients
        importances = None
        try:
            importances = model.feature_importances_
        except AttributeError:
            try:
                importances = model.coef_[0]
            except AttributeError:
                pass

        # Determine top contributing factors
        top_factors = []
        if importances is not None:
            top_indices = sorted(range(len(importances)), key=lambda i: abs(importances[i]), reverse=True)[:3]
            top_factors = [(feature_names[i], round(importances[i], 4)) for i in top_indices]

        return render_template(
            "result.html", 
            prediction="Likely to churn" if prediction == 1 else "Not likely to Churn",
            top_factors=top_factors
        )
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == "__main__":
    app.run(debug=True)