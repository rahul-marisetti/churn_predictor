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
    return render_template("index.html")  #index page

@app.route('/analysis', methods=['GET'])
def analysis():
    return render_template("analysis.html")  #analysis page

@app.route('/predict', methods=['POST'])
def predict():
    try:
        #input
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

        #predict using the model
        input_array = np.array(input_features).reshape(1, -1)
        prediction = model.predict(input_array)[0]

        #prediction result
        result = "Customer is likely to churn." if prediction == 1 else "Customer is likely to stay."
        return redirect(url_for('result', prediction=result))
    
    except Exception as e:
        return f"Error: {e}"

@app.route('/result')
def result():
    prediction = request.args.get("prediction", "No prediction available.")
    return render_template("result.html", result=prediction)

if __name__ == "__main__":
    app.run(debug=True)
