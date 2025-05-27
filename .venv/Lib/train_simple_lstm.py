import pandas as pd
import numpy as np
import math
import datetime as dt
import yfinance as yf
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score 
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class SimpleLSTMTrainer:
    def __init__(self, symbol='AAPL', period='5y', time_step=15):
        self.symbol = symbol
        self.period = period
        self.time_step = time_step
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.data = None
        
    def download_and_prepare_data(self):
        print(f"Загрузка данных для {self.symbol}...")
        ticker = yf.Ticker(self.symbol)
        data = ticker.history(period=self.period, interval='1d')
        
        if data.empty:
            raise ValueError(f"Не удалось загрузить данные для {self.symbol}")

        self.data = pd.DataFrame()
        self.data['date'] = data.index
        self.data['open'] = data['Open'].values
        self.data['high'] = data['High'].values
        self.data['low'] = data['Low'].values
        self.data['close'] = data['Close'].values
        self.data['volume'] = data['Volume'].values

        self.data.sort_values(by='date', inplace=True)
        self.data.reset_index(drop=True, inplace=True)
        
        print(f"Загружено {len(self.data)} записей")
        print(f"Начальная дата: {self.data.iloc[0]['date']}")
        print(f"Конечная дата: {self.data.iloc[-1]['date']}")
        print(f"Период: {self.data.iloc[-1]['date'] - self.data.iloc[0]['date']}")
        
        return self.data
    
    def prepare_close_data(self):
        closedf = self.data[['date', 'close']].copy()
        print(f"Форма DataFrame с ценами закрытия: {closedf.shape}")
        self.close_stock = closedf.copy()
        close_values = closedf[['close']].values
        self.closedf_scaled = self.scaler.fit_transform(close_values)
        print(f"Форма нормализованных данных: {self.closedf_scaled.shape}")
        
        return self.closedf_scaled
    
    def split_data(self, train_ratio=0.65):
        training_size = int(len(self.closedf_scaled) * train_ratio)
        test_size = len(self.closedf_scaled) - training_size
        
        self.train_data = self.closedf_scaled[0:training_size, :]
        self.test_data = self.closedf_scaled[training_size:len(self.closedf_scaled), :1]
        
        print(f"Размер обучающих данных: {self.train_data.shape}")
        print(f"Размер тестовых данных: {self.test_data.shape}")
        print(f"Соотношение train/test: {train_ratio:.0%}/{1-train_ratio:.0%}")
        
        return self.train_data, self.test_data
    
    def create_dataset(self, dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - time_step - 1):
            a = dataset[i:(i + time_step), 0]
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)
    
    def prepare_sequences(self):
        self.X_train, self.y_train = self.create_dataset(self.train_data, self.time_step)
        self.X_test, self.y_test = self.create_dataset(self.test_data, self.time_step)
        
        print(f"X_train: {self.X_train.shape}")
        print(f"y_train: {self.y_train.shape}")
        print(f"X_test: {self.X_test.shape}")
        print(f"y_test: {self.y_test.shape}")

        self.X_train = self.X_train.reshape(self.X_train.shape[0], self.X_train.shape[1], 1)
        self.X_test = self.X_test.reshape(self.X_test.shape[0], self.X_test.shape[1], 1)
        
        print(f"После reshape:")
        print(f"X_train: {self.X_train.shape}")
        print(f"X_test: {self.X_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def build_model(self, lstm_units=32):
        tf.keras.backend.clear_session()
        
        model = Sequential([
            LSTM(lstm_units, return_sequences=True, input_shape=(self.time_step, 1)),
            Dropout(0.2),

            LSTM(lstm_units, return_sequences=True),
            Dropout(0.2),

            LSTM(lstm_units),
            Dropout(0.2),

            Dense(1)
        ])

        model.compile(
            loss='mean_squared_error',
            optimizer='adam',
            metrics=['mae']
        )
        
        print("Архитектура модели:")
        model.summary()
        
        self.model = model
        return model
    
    def train_model(self, epochs=200, batch_size=5, validation_split=0.1):
        print(f"\nНачинаем обучение модели...")
        print(f"Эпохи: {epochs}, Размер батча: {batch_size}")
        
        # Callbacks для улучшения обучения
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        )

        os.makedirs('models', exist_ok=True)

        history = self.model.fit(
            self.X_train, self.y_train,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        self.history = history
        return history
    
    def make_predictions(self):
        print("\nСоздание предсказаний...")

        self.train_predict = self.model.predict(self.X_train)
        self.test_predict = self.model.predict(self.X_test)
        
        print(f"Форма предсказаний: train={self.train_predict.shape}, test={self.test_predict.shape}")

        self.train_predict = self.scaler.inverse_transform(self.train_predict)
        self.test_predict = self.scaler.inverse_transform(self.test_predict)
        self.original_ytrain = self.scaler.inverse_transform(self.y_train.reshape(-1, 1))
        self.original_ytest = self.scaler.inverse_transform(self.y_test.reshape(-1, 1))
        
        return self.train_predict, self.test_predict
    
    def evaluate_model(self):
        print("\n" + "="*80)
        print("ОЦЕНКА КАЧЕСТВА МОДЕЛИ")
        print("="*80)

        train_rmse = math.sqrt(mean_squared_error(self.original_ytrain, self.train_predict))
        train_mse = mean_squared_error(self.original_ytrain, self.train_predict)
        train_mae = mean_absolute_error(self.original_ytrain, self.train_predict)
        train_r2 = r2_score(self.original_ytrain, self.train_predict)

        test_rmse = math.sqrt(mean_squared_error(self.original_ytest, self.test_predict))
        test_mse = mean_squared_error(self.original_ytest, self.test_predict)
        test_mae = mean_absolute_error(self.original_ytest, self.test_predict)
        test_r2 = r2_score(self.original_ytest, self.test_predict)

        test_mape = np.mean(np.abs((self.original_ytest - self.test_predict) / self.original_ytest)) * 100

        actual_direction = np.sign(np.diff(self.original_ytest.flatten()))
        predicted_direction = np.sign(np.diff(self.test_predict.flatten()))
        direction_accuracy = np.mean(actual_direction == predicted_direction)
        
        print("ОБУЧАЮЩИЕ ДАННЫЕ:")
        print(f"RMSE: {train_rmse:.6f}")
        print(f"MSE:  {train_mse:.6f}")
        print(f"MAE:  {train_mae:.6f}")
        print(f"R²:   {train_r2:.6f}")
        
        print("\nТЕСТОВЫЕ ДАННЫЕ:")
        print(f"RMSE: {test_rmse:.6f}")
        print(f"MSE:  {test_mse:.6f}")
        print(f"MAE:  {test_mae:.6f}")
        print(f"R²:   {test_r2:.6f}")
        print(f"MAPE: {test_mape:.2f}%")
        print(f"Точность направления: {direction_accuracy:.2%}")

        print(f"\n📊 СРАВНЕНИЕ С ПРЕДЫДУЩИМИ РЕЗУЛЬТАТАМИ:")
        prev_rmse = 36.24
        prev_mae = 34.57
        prev_mape = 15.05
        
        if test_rmse < prev_rmse:
            rmse_improvement = ((prev_rmse - test_rmse) / prev_rmse) * 100
            print(f"✅ RMSE улучшение: {rmse_improvement:.1f}% (было {prev_rmse:.2f}, стало {test_rmse:.2f})")
        else:
            rmse_degradation = ((test_rmse - prev_rmse) / prev_rmse) * 100
            print(f"❌ RMSE ухудшение: {rmse_degradation:.1f}% (было {prev_rmse:.2f}, стало {test_rmse:.2f})")
        
        if test_mae < prev_mae:
            mae_improvement = ((prev_mae - test_mae) / prev_mae) * 100
            print(f"✅ MAE улучшение: {mae_improvement:.1f}% (было {prev_mae:.2f}, стало {test_mae:.2f})")
        else:
            mae_degradation = ((test_mae - prev_mae) / prev_mae) * 100
            print(f"❌ MAE ухудшение: {mae_degradation:.1f}% (было {prev_mae:.2f}, стало {test_mae:.2f})")
        
        if test_mape < prev_mape:
            mape_improvement = ((prev_mape - test_mape) / prev_mape) * 100
            print(f"✅ MAPE улучшение: {mape_improvement:.1f}% (было {prev_mape:.2f}%, стало {test_mape:.2f}%)")
        else:
            mape_degradation = ((test_mape - prev_mape) / prev_mape) * 100
            print(f"❌ MAPE ухудшение: {mape_degradation:.1f}% (было {prev_mape:.2f}%, стало {test_mape:.2f}%)")
        
        return {
            'train_rmse': train_rmse,
            'train_mse': train_mse,
            'train_mae': train_mae,
            'train_r2': train_r2,
            'test_rmse': test_rmse,
            'test_mse': test_mse,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'test_mape': test_mape,
            'direction_accuracy': direction_accuracy
        }
    
    def plot_results(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        axes[0, 0].plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        axes[0, 0].plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].scatter(self.original_ytest, self.test_predict, alpha=0.6, s=20)
        min_val = min(self.original_ytest.min(), self.test_predict.min())
        max_val = max(self.original_ytest.max(), self.test_predict.max())
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        axes[0, 1].set_xlabel('Actual Prices')
        axes[0, 1].set_ylabel('Predicted Prices')
        axes[0, 1].set_title('Test: Predictions vs Actual', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)

        train_plot = np.empty_like(self.closedf_scaled)
        train_plot[:, :] = np.nan
        train_plot[self.time_step:len(self.train_predict) + self.time_step, :] = self.train_predict
        
        test_plot = np.empty_like(self.closedf_scaled)
        test_plot[:, :] = np.nan
        test_plot[len(self.train_predict) + (self.time_step * 2) + 1:len(self.closedf_scaled) - 1, :] = self.test_predict

        original_data = self.scaler.inverse_transform(self.closedf_scaled)
        
        axes[1, 0].plot(original_data, label='Original Data', linewidth=2)
        axes[1, 0].plot(train_plot, label='Training Predictions', linewidth=2)
        axes[1, 0].plot(test_plot, label='Test Predictions', linewidth=2)
        axes[1, 0].set_title('Full Time Series Predictions', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Price')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        test_errors = self.original_ytest.flatten() - self.test_predict.flatten()
        axes[1, 1].hist(test_errors, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Test Error Distribution', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Prediction Error')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].axvline(x=0, color='red', linestyle='--')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'models/simple_lstm_results_{self.symbol}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self):
        self.model.save(f'models/simple_lstm_{self.symbol}.h5')
        joblib.dump(self.scaler, f'models/simple_lstm_scaler_{self.symbol}.pkl')

        model_params = {
            'symbol': self.symbol,
            'period': self.period,
            'time_step': self.time_step,
            'model_type': 'simple_lstm',
            'data_shape': self.closedf_scaled.shape
        }
        joblib.dump(model_params, f'models/simple_lstm_params_{self.symbol}.pkl')
        
        print(f"\nМодель сохранена в папке models/")
    
    def run_full_training(self, epochs=200, batch_size=5, lstm_units=32):
        try:
            print("🚀 ЗАПУСК ПРОСТОЙ LSTM МОДЕЛИ")
            print("="*50)
            self.download_and_prepare_data()
            self.prepare_close_data()
            self.split_data()
            self.prepare_sequences()
            self.build_model(lstm_units)
            self.train_model(epochs, batch_size)
            self.make_predictions()
            results = self.evaluate_model()
            self.plot_results()
            self.save_model()
            
            print("\n✅ ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
            return results
            
        except Exception as e:
            print(f"❌ Ошибка при обучении: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    SYMBOL = 'AAPL'
    PERIOD = '5y'
    TIME_STEP = 15
    EPOCHS = 200
    BATCH_SIZE = 5
    LSTM_UNITS = 32
    
    print(f"Параметры обучения:")
    print(f"Символ: {SYMBOL}")
    print(f"Период: {PERIOD}")
    print(f"Временные шаги: {TIME_STEP}")
    print(f"Эпохи: {EPOCHS}")
    print(f"Размер батча: {BATCH_SIZE}")
    print(f"LSTM нейроны: {LSTM_UNITS}")

    trainer = SimpleLSTMTrainer(
        symbol=SYMBOL,
        period=PERIOD,
        time_step=TIME_STEP
    )
    
    results = trainer.run_full_training(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lstm_units=LSTM_UNITS
    )
    
    if results:
        print(f"\n🎯 ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ:")
        print(f"Test RMSE: {results['test_rmse']:.6f}")
        print(f"Test MAE: {results['test_mae']:.6f}")
        print(f"Test MAPE: {results['test_mape']:.2f}%")
        print(f"Test R²: {results['test_r2']:.6f}")
        print(f"Точность направления: {results['direction_accuracy']:.2%}")

if __name__ == "__main__":
    main() 