import onnxruntime as ort
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


class Analyzer:
    def __init__(self):
        self.model_path = "model/model.onnx"

        self.ort_session = ort.InferenceSession(self.model_path)

        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name = self.ort_session.get_outputs()[0].name

        self.start_date = "2023-01-01"
        self.end_date = "2025-03-27"

    def calculate_features(self, df):
        df['Returns'] = df['Close'].pct_change()
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()

        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))

        df.dropna(inplace=True)
        return df

    def predict_price(self, row):
        input_data = np.array([[
            row['Open'], row['High'], row['Low'], row['Volume'],
            row['SMA_20'], row['SMA_50'], row['RSI']
        ]], dtype=np.float32).reshape(1, -1)

        pred = self.ort_session.run([self.output_name], {self.input_name: input_data})[0][0][0]
        return pred