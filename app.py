import numpy as np
from flask import Flask, request, render_template
import pickle
import sklearn.feature_extraction.text

#Create an app object using the Flask class.
app = Flask(__name__)
#Load the trained model. (Pickle file)
model = pickle.load(open('models/model.pkl', 'rb'))
#Define the route to be home.
#use the route() decorator to tell Flask what URL should trigger our function.
@app.route('/')
def home():
    return render_template('index.html')
#You can use the methods argument of the route() decorator to handle different HTTP methods.#GET: A GET message is send, and the server returns data
#POST: Used to send HTML form data to the server.
@app.route('/predict',methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()] #Convert string inputs to float.
    features = [np.array(int_features)]  #Convert to the form [[a, b]] for input to the model
    prediction = model.predict(features)  # features Must be in the form [[a, b]]
    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Percent with heart disease is {}'.format(output))

if __name__ == "__main__":
    app.run()