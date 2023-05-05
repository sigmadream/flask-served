from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route('/api/predict', methods=['POST'])
def predict():
    if classifier:
        try:
            content = request.json
            print(content)
            Sex = content['Sex']
            Age = content['Age']
            Pclass = content['Pclass']
            input = [[Sex, Age, Pclass]]
            print('raw: ', input)
            input_ct = ct.transform(input)
            input_sc = sc.transform(input_ct)
            prediction = classifier.predict(input_sc)
            return jsonify({'prediction': str(prediction)})
        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        print('Model not found')
        return('Model not found')

if __name__ == '__main__':
    classifier = joblib.load("./model/model.pkl")
    sc = joblib.load("./model/sc.pkl")
    ct = joblib.load("./model/ct.pkl")
    print ('Model loaded')
    app.run(debug=True)