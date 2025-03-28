## UNUSED, WILL USE XGBoost

import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

def preprocess_data(df: pd.DataFrame, threshold: int):
    df['target'] = (df['points'] > threshold).astype(int)
    
    X = df.drop(columns=['target', 'game_date']).values
    y = df['target'].values

    scaler = StandardScaler()

    X = scaler.fit_transform(X)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).view(-1, 1)

def train_model(df, threshold, epochs=10, batch_size=32, learning_rate=0.001):
    X, y = preprocess_data(df, threshold=threshold)

    X_train, X_test, y_train, y_test = train_test_split(X.numpy(), y.numpy(), test_size=0.2, random_state=42)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    input_dim = X_train.shape[1]

    model = Net(input_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device=device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        X_train = X_train.to(device)
        y_train = y_train.to(device)
        preds = model(X_train)

        loss = criterion(preds, y_train)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 1 == 0:
            print(f'Epoch: {epoch + 1} \nLoss: {loss.item():.4f}')
    
    model.eval()

    with torch.no_grad():
        X_test = X_test.to(device)
        y_test = y_test.to(device)
        y_pred = model(X_test)
        y_pred = (y_pred > 0.5).float()

        accuracy = (y_pred == y_test).float().mean()

        print(f'Accuracy on test data: {accuracy.item():.4f}')
    
    return model

threshold = 20

df = pd.read_csv('./cleaned/final.csv')

model = train_model(df, threshold)