from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load your pre-trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define your feature names
feature_names = ['Age', 'Sex (1 = male; 0 = female) ','Chest Pain Type (0,1,2,3)','Resting Blood Pressure','Serum Cholestoral in mg/dl','Fasting Blood Sugar > 120 mg/dl (1 = true; 0 = false)','Resting Electrocardiographic Results (0,1,2)','Maximum Heart Rate Achieved','Exercise Induced Angina (1 = yes; 0 = no)','Oldpeak = ST depression induced by exercise relative to rest','The slope of the peak exercise ST segment','Number of major vessels (0-3) colored by flourosopy','Thal : 1 = normal; 2 = fixed defect; 3 = reversable defect']

@app.route('/')
def home():
    return render_template('index.html', feature_names=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input from the form
        features = [float(request.form[f]) for f in feature_names]

        # Perform prediction using the loaded model
        prediction = model.predict([features])

        # Display the prediction on a new page or in the same page
        return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
