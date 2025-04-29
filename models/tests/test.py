import onnxruntime as ort
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

model_path = "model.onnx"

ort_session = ort.InferenceSession(model_path)

input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name
print(f"Входное имя: {input_name}, Выходное имя: {output_name}")

ticker = "GOOGL"
start_date = "2023-01-01"
end_date = "2025-03-27"

data = yf.download(ticker, start=start_date, end=end_date)
data.head()


def calculate_features(df):
    df['Returns'] = df['Close'].pct_change()
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()

    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))

    df.dropna(inplace=True)
    return df


data = calculate_features(data)


def predict_price(row):
    input_data = np.array([[
        row['Open'], row['High'], row['Low'], row['Volume'],
        row['SMA_20'], row['SMA_50'], row['RSI']
    ]], dtype=np.float32).reshape(1, -1)

    # Прогнозирование
    pred = ort_session.run([output_name], {input_name: input_data})[0][0][0]
    print(f"Тестовое предсказание: {pred:.2f}")
    return pred


data['Predicted'] = data.apply(predict_price, axis=1)

plt.figure(figsize=(14, 6))
plt.plot(data.index, data['Close'], label='Фактическая цена', color='blue')
plt.plot(data.index, data['Predicted'], label='Прогноз', color='orange', linestyle='--')
plt.title(f"Прогнозирование цен {ticker}")
plt.xlabel('Дата')
plt.ylabel('Цена ($)')
plt.legend()
plt.grid()
plt.show()

