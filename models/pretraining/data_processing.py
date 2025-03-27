import pandas as pd
import numpy as np


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    # Убедимся, что данные отсортированы по дате
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)

    # Простая доходность
    df['Returns'] = df['Close'].pct_change()

    # Скользящие средние
    df['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['SMA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()

    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Заполним пропуски, если они есть
    df.fillna(0, inplace=True)
    return df


# Загрузка данных и расчет признаков
data = pd.read_csv("apple_stock.csv")
data = calculate_features(data)

# Сохраним обработанные данные
data.to_csv("apple_stock_processed.csv", index=False)