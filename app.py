from flask import Flask, request,jsonify,render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

model = pickle.load(open("C:\\Users\\hp\\PycharmProjects\\flask_project\\model1.pkl", 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]

    return render_template('index.html', prediction_text='Expense = {}'.format(output))

if __name__ == "__main__":
    app.run(host="127.0.0.3",debug=True)







