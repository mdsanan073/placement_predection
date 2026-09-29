import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


class PlacementNN(nn.Module):
    def __init__(self, input_features=8):
        super(PlacementNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    semesters = [np.random.uniform(50.0, 100.0, num_samples) for _ in range(6)]
    dsa = np.random.uniform(30.0, 100.0, num_samples)
    comm = np.random.uniform(3.0, 10.0, num_samples)

    X = np.column_stack((*semesters, dsa, comm))
    avg_sem = sum(semesters) / 6.0
    score = (avg_sem / 10 * 0.4) + (dsa / 10 * 0.4) + (comm * 0.2)
    y = (score > np.percentile(score, 40)).astype(float)

    # Add noise
    noise_idx = np.random.choice(num_samples, size=int(num_samples * 0.1), replace=False)
    y[noise_idx] = 1 - y[noise_idx]
    return X, y


if __name__ == "__main__":
    print("Generating data...")
    X, y = generate_synthetic_data(1500)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    os.makedirs('saved_models', exist_ok=True)
    joblib.dump(scaler, 'saved_models/scaler.save')

    # 1. Train scikit-learn models
    print("Training classical ML models...")
    sklearn_models = {
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier(n_estimators=100),
        'SVM': SVC(probability=True),
        'Decision Tree': DecisionTreeClassifier(),
        'Naive Bayes': GaussianNB(),
        'KNN': KNeighborsClassifier(),
        'Gradient Boosting': GradientBoostingClassifier()
    }

    for name, model in sklearn_models.items():
        model.fit(X_train_scaled, y_train)
    joblib.dump(sklearn_models, 'saved_models/sklearn_models.save')

    # 2. Train PyTorch Model (utilizing GPU if available)
    print("Training PyTorch Neural Network...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nn_model = PlacementNN().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(nn_model.parameters(), lr=0.01)

    X_tensor = torch.FloatTensor(X_train_scaled).to(device)
    y_tensor = torch.FloatTensor(y_train).view(-1, 1).to(device)

    for epoch in range(300):
        optimizer.zero_grad()
        loss = criterion(nn_model(X_tensor), y_tensor)
        loss.backward()
        optimizer.step()

    torch.save(nn_model.state_dict(), 'saved_models/placement_model.pth')
    print("All models successfully trained and saved.")