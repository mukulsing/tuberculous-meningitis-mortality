
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    features = [
        float(data.get('gcs',0)),
        float(data.get('hydrocephalus',0)),
        float(data.get('age',0)),
        float(data.get('csf_glucose',0)),
        float(data.get('infarction',0)),
        float(data.get('duration',0)),
        float(data.get('seizures',0)),
        float(data.get('consciousness',0)),
        float(data.get('steroids',0))
    ]
    
    pred = model.predict([features])[0]
    prob = model.predict_proba([features])[0][1]
    
    return jsonify({"mortality": int(pred), "probability": float(prob)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
