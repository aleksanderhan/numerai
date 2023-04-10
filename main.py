import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from numerapi import NumerAPI
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam
import optuna

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class ODEFunc(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ODEFunc, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(0.5)

    def forward(self, t, y):
        y = self.norm1(F.relu(self.fc1(y)))
        y = self.dropout(y)
        y = self.norm2(F.relu(self.fc2(y)))
        y = self.dropout(y)
        y = self.fc3(y)
        return y

def runge_kutta_4(func, y0, t, h):
    k1 = func(t, y0)
    k2 = func(t + h / 2, y0 + h * k1 / 2)
    k3 = func(t + h / 2, y0 + h * k2 / 2)
    k4 = func(t + h, y0 + h * k3)
    return y0 + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

from torchdiffeq import odeint

class DeepODE(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DeepODE, self).__init__()
        self.ode_func = ODEFunc(input_dim, hidden_dim)

    def forward(self, y0, t):
        y = odeint(self.ode_func, y0, t, method='dopri5')  # 'dopri5' is the Dormand-Prince method
        return y[-1]  # Return the last state



def preprocess_data(df, feats):
    # Select the specified features and target
    features = df[feats]
    targets = df["target"]

    # Scale features using StandardScaler
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Convert features and targets to PyTorch tensors
    features_tensor = torch.tensor(features, dtype=torch.float32)
    targets_tensor = torch.tensor(targets.values, dtype=torch.float32)

    return features_tensor, targets_tensor

import json

def load_numerai_data():
    napi = NumerAPI()

    # Download datasets
    napi.download_dataset("v4.1/train_int8.parquet", dest_path="train_int8.parquet")
    napi.download_dataset("v4.1/validation_int8.parquet", dest_path="validation_int8.parquet")
    
    # Download and load features
    napi.download_dataset("v4.1/features.json", dest_path="features.json")
    with open("features.json", "rb") as f:
        feats = json.load(f)["feature_sets"]["medium"]

    # Load datasets using pandas
    train_df = pd.read_parquet("train_int8.parquet")
    validation_df = pd.read_parquet("validation_int8.parquet")

    return train_df, validation_df, feats

def objective(trial, input_dim):
    # Hyperparameters to optimize
    hidden_dim = trial.suggest_int("hidden_dim", 32, 128)
    h = trial.suggest_float("h", 0.01, 0.1)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
    epochs = trial.suggest_int("epochs", 50, 150)

    # Create DeepODE model
    model = DeepODE(input_dim, hidden_dim).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X_train, torch.tensor([0.0, 10.0], device=device))
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

    # Compute validation loss
    with torch.no_grad():
        y_val_pred = model(X_val, torch.tensor([0.0, 10.0], device=device))
        val_loss = criterion(y_val_pred, y_val)

    return val_loss.item()

train_df, validation_df, feats = load_numerai_data()

X_train, y_train = preprocess_data(train_df, feats)
X_val, y_val = preprocess_data(validation_df, feats)

X_train, X_val = X_train.to(device), X_val.to(device)
y_train, y_val = y_train.to(device), y_val.to(device)


study = optuna.create_study(direction="minimize")
study.optimize(lambda trial: objective(trial, len(feats)), n_trials=50)

print("Best trial:")
print(f"  Value: {study.best_trial.value}")
print(f"  Params: {study.best_trial.params}")

best_params = study.best_trial.params


input_dim = X_train.shape[1]
hidden_dim = best_params["hidden_dim"]
h = best_params["h"]
learning_rate = best_params["learning_rate"]
epochs = best_params["epochs"]

model = DeepODE(input_dim, hidden_dim, h).to(device)
optimizer = Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model(X_train, t=10)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}: Loss {loss.item()}")