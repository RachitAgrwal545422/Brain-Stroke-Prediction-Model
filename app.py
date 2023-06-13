# import All the necessary python packages
from flask import Flask, render_template, request
import numpy as np
import pickle as pk

# Initiate the app
app = Flask(__name__)
# Take the first request and render the home page


@app.route("/")
def index():
    return render_template("home.html")


# Render the submit request
@app.route("/result", methods=['POST', 'GET'])
def result():
    # Collect the information from the form
    gender = int(request.form['gender'])
    age = int(request.form['age'])
    hypertension = int(request.form['hypertension'])
    heart_disease = int(request.form['heart_disease'])
    ever_married = int(request.form['ever_married'])
    work_type = int(request.form['work_type'])
    Residence_type = int(request.form['Residence_type'])
    avg_glucose_level = float(request.form['avg_glucose_level'])
    bmi = float(request.form['bmi'])
    smoking_status = int(request.form['smoking_status'])

    # Convert the information to the numpy array
    x = np.array([gender, age, hypertension, heart_disease, ever_married, work_type,
                 Residence_type, avg_glucose_level, bmi, smoking_status]).reshape(1, -1)

    # Load the classfier to transform the input into a standard or normalized form
    with open('model/scaler.pkl', 'rb') as f:
        scaler = pk.load(f)
    x = scaler.transform(x)

    # Load the model to predict the results
    with open('model/lr.sav', 'rb') as f:
        model = pk.load(f)
    Y_pred = model[0].predict(x)

    # render the page based on the value of model prediction
    if Y_pred == 0:
        return render_template('nostroke.html')
    else:
        return render_template('stroke.html')


# basic template to run the application
if __name__ == "__main__":
    app.run(debug=True, port=7384)
