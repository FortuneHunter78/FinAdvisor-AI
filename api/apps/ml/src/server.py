from typing import Union
from fastapi import FastAPI
import yfinance as yf


from src.dto.dto import TickerInput
from src.analyzer.analyzer import Analyzer
from src.model_loader import ModelLoader
from src.s3.s3 import MinioAsyncStorage

app = FastAPI()

# Initialize MinIO and ModelLoader
storage = MinioAsyncStorage(
    endpoint_url="http://minio:9000",
    access_key="admin",
    secret_key="password",
    bucket="weights",
)

model_loader = ModelLoader(storage, model_key="model.onnx")

@app.on_event("startup")
async def startup_event():
    await model_loader.load_model()
    print("✅ Model loaded from S3 on startup")


@app.get("/")
def read_root():

    return {"message": "OK"}

@app.post("/predict")
def predict(input_data: TickerInput):
    analyzer = Analyzer()
    ticker = input_data.ticker

    # Hardcode
    start_date = "2023-01-01"
    end_date = "2025-03-27"

    data = yf.download(ticker, start=start_date, end=end_date)
    data = analyzer.calculate_features(data)
    data['Predicted'] = data.apply(analyzer.predict_price, axis=1)

    return {"output": data["Predicted"].tolist()}
