import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd

data = pd.read_csv("apple_stock_processed.csv")

# Целевая переменная: цена закрытия завтрашнего дня
data['Target'] = data['Close'].shift(-1)
data.dropna(inplace=True)

# Признаки и целевая переменная
features = ['Open', 'High', 'Low', 'Volume', 'SMA_20', 'SMA_50', 'RSI']
X = data[features]
y = data['Target']

# Разделение данных (без перемешивания!)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Создание модели
model = lgb.LGBMRegressor(
    n_estimators=200,
    learning_rate=0.05,
    random_state=42
)

# Обучение
model.fit(X_train, y_train)

# Прогноз
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"MAE: {mae:.2f}")

# Сохранение модели
model.booster_.save_model("model.txt")