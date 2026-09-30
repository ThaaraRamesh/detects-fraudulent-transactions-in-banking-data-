from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import random
import datetime
import os

app = FastAPI(title="Fraud Detection API")
# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and scaler (if they exist)
model_path = 'model.joblib'
scaler_path = 'scaler.joblib'
logs_path = 'mock_logs.joblib'

class Transaction(BaseModel):
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    is_international: int

@app.get("/api/transactions")
def get_transactions():
    if os.path.exists(logs_path):
        logs = joblib.load(logs_path)
        # Sort by timestamp descending
        logs.sort(key=lambda x: x['timestamp'], reverse=True)
        return {"transactions": logs}
    return {"transactions": []}

@app.get("/")
def read_root():
    return FileResponse("../frontend/index.html")

app.mount("/static", StaticFiles(directory="../frontend"), name="static")

@app.post("/api/predict")
def predict_fraud(transaction: Transaction):
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise HTTPException(status_code=500, detail="Model not trained yet. Run train_model.py first.")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Prepare data for prediction
    df = pd.DataFrame([{
        'amount': transaction.amount,
        'oldbalanceOrg': transaction.oldbalanceOrg,
        'newbalanceOrig': transaction.newbalanceOrig,
        'is_international': transaction.is_international
    }])
    
    # Scale features
    df_scaled = scaler.transform(df)
    
    # Predict
    prob = model.predict_proba(df_scaled)[0][1]
    is_fraud = int(model.predict(df_scaled)[0])
    
    # Create new transaction log entry
    txn_id = f'TXN{random.randint(10000, 99999)}'
    new_txn = {
        'id': txn_id,
        'timestamp': datetime.datetime.now().isoformat(),
        'amount': transaction.amount,
        'oldbalanceOrg': transaction.oldbalanceOrg,
        'newbalanceOrig': transaction.newbalanceOrig,
        'is_international': transaction.is_international,
        'is_fraud': is_fraud,
        'fraud_prob': prob
    }
    
    # Save back to logs
    logs = joblib.load(logs_path) if os.path.exists(logs_path) else []
    logs.append(new_txn)
    joblib.dump(logs, logs_path)
    
    return {
        "transaction_id": txn_id,
        "is_fraud": is_fraud,
        "probability": prob
    }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
