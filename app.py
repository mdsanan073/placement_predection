import os
import subprocess
import sys
from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import numpy as np
import joblib

app = Flask(__name__)


class PlacementNN(nn.Module):
    def __init__(self, input_features=8):
        super(PlacementNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 1), nn.Sigmoid()
        )

    def forward(self, x): return self.network(x)


# Auto-train logic updated to check for both files
if not os.path.exists('saved_models/scaler.save') or not os.path.exists('saved_models/sklearn_models.save'):
    print("Model files missing or outdated. Automatically running train_model.py...")
    subprocess.run([sys.executable, "train_model.py"], check=True)

# Load Models
scaler = joblib.load('saved_models/scaler.save')
sklearn_models = joblib.load('saved_models/sklearn_models.save')
torch_model = PlacementNN()
torch_model.load_state_dict(torch.load('saved_models/placement_model.pth', weights_only=True, map_location='cpu'))
torch_model.eval()


@app.route('/')
def index(): return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict_api():
    try:
        data = request.get_json(force=True)
        features = [float(data.get(f, 0)) for f in
                    ['sem1', 'sem2', 'sem3', 'sem4', 'sem5', 'sem6', 'dsa_score', 'comm_skills']]
        selected_model = data.get('model_type', 'Neural Network')

        scaled_features = scaler.transform(np.array([features]))

        # Route to correct model
        if selected_model == 'Neural Network':
            with torch.no_grad():
                prob_raw = torch_model(torch.FloatTensor(scaled_features)).item()
        else:
            prob_raw = sklearn_models[selected_model].predict_proba(scaled_features)[0][1]

        probability = round(prob_raw * 100, 2)

        return jsonify({
            'success': True,
            'probability': probability,
            'model_used': selected_model,
            'status': 'High Chance' if probability >= 50 else 'May Need Improvement'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)