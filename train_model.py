import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import random
import datetime

def generate_data(n=1000):
    np.random.seed(42)
    # Features: amount, oldbalanceOrg, newbalanceOrig, is_international
    amount = np.random.exponential(scale=500, size=n)
    oldbalanceOrg = np.random.uniform(0, 100000, size=n)
    newbalanceOrig = oldbalanceOrg - amount
    is_international = np.random.choice([0, 1], size=n, p=[0.9, 0.1])
    
    # Target: Fraud (1) or Legitimate (0)
    # Higher amounts and international transfers have higher fraud likelihood
    z = (amount - 500) / 500 + is_international * 2
    prob = 1 / (1 + np.exp(-z))
    is_fraud = (np.random.rand(n) < prob).astype(int)
    
    # Adjust some data to make the dataset slightly more realistic
    is_fraud[amount < 50] = 0
    
    data = pd.DataFrame({
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'is_international': is_international,
        'is_fraud': is_fraud
    })
    
    return data

def train_and_save_model():
    print("Generating synthetic data...")
    df = generate_data(5000)
    
    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Training Logistic Regression model...")
    model = LogisticRegression(class_weight='balanced')
    model.fit(X_scaled, y)
    
    print(f"Accuracy: {model.score(X_scaled, y):.4f}")
    
    # Save the model and scaler
    joblib.dump(model, 'model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    
    # Generate some mock transactions to display in the viewer initially
    mock_transactions = []
    for _ in range(50):
        t_amount = round(random.expovariate(1/500), 2)
        t_old = round(random.uniform(100, 50000), 2)
        mock_transactions.append({
            'id': f'TXN{random.randint(10000, 99999)}',
            'timestamp': (datetime.datetime.now() - datetime.timedelta(minutes=random.randint(0, 1000))).isoformat(),
            'amount': t_amount,
            'oldbalanceOrg': t_old,
            'newbalanceOrig': t_old - t_amount,
            'is_international': random.choice([0, 1])
        })
    joblib.dump(mock_transactions, 'mock_logs.joblib')
    print("Saved mock logs.")

if __name__ == '__main__':
    train_and_save_model()
