import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import argparse
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report
import joblib
import json
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

sys.path.append('.venv/Lib')

from combined_model import CombinedPredictionModel, SimpleLSTMModel
from update_data import update_stock_data, get_news
from macro_data import MacroDataLoader
from main import get_company_info_auto, get_news_headlines, get_global_political_news, is_russian_ticker, get_moex_data

class ModelBacktester:
    def __init__(self, 
                 model_path: str = "/models",
                 lookback_period: int = 60,
                 test_period: int = 30,
                 prediction_horizon: int = 1,
                 use_news: bool = False,  # Отключаем новости по умолчанию для стабильности
                 use_macro: bool = False):

        self.model_path = model_path
        self.lookback_period = lookback_period
        self.test_period = test_period
        self.prediction_horizon = prediction_horizon
        self.use_news = use_news
        self.use_macro = use_macro

        self.model = None
        self.macro_loader = MacroDataLoader() if use_macro else None

        self.results = {
            'ticker': [],
            'date': [],
            'actual_price': [],
            'predicted_price': [],
            'actual_direction': [],
            'predicted_direction': [],
            'confidence': [],
            'price_error': [],
            'price_error_pct': [],
            'is_correct_direction': []
        }

        self.stats = {}
    
    def find_model_files(self, ticker: str) -> Dict[str, str]:
        model_files = {
            'h5_file': None,
            'params_file': None,
            'scaler_file': None
        }
        
        if not os.path.exists(self.model_path):
            print(f"Путь к модели не существует: {self.model_path}")
            return model_files
            
        files = os.listdir(self.model_path)

        for file in files:
            if ticker.upper() in file.upper():
                if file.endswith('.h5'):
                    model_files['h5_file'] = os.path.join(self.model_path, file)
                elif 'params' in file and file.endswith('.pkl'):
                    model_files['params_file'] = os.path.join(self.model_path, file)
                elif 'scaler' in file and file.endswith('.pkl'):
                    model_files['scaler_file'] = os.path.join(self.model_path, file)
        
        return model_files
    
    def load_model_for_ticker(self, ticker: str) -> bool:
        try:
            model_files = self.find_model_files(ticker)
            
            if all(model_files.values()):
                print(f"Найдены файлы модели для {ticker}:")
                for key, path in model_files.items():
                    print(f"  {key}: {os.path.basename(path)}")

                self.model = SimpleLSTMModel(model_path=self.model_path, symbol=ticker)
                success = self.model.load_models(self.model_path)
                
                if success:
                    print(f"Модель для {ticker} успешно загружена")
                    return True
                else:
                    print(f"Ошибка загрузки модели для {ticker}")
                    return False
            else:
                print(f"Не найдены все необходимые файлы модели для {ticker}")
                print(f"Найдено: {model_files}")

                self.model = SimpleLSTMModel()
                print(f"Используется fallback модель для {ticker}")
                return False
                
        except Exception as e:
            print(f"Ошибка при загрузке модели для {ticker}: {e}")
            self.model = SimpleLSTMModel()
            return False
    
    def load_stock_data(self, ticker: str, end_date: datetime = None, is_moex: bool = False) -> pd.DataFrame:
        if end_date is None:
            end_date = datetime.now()
        
        start_date = end_date - timedelta(days=self.lookback_period + self.test_period + 10)
        
        try:
            if is_moex:
                print(f"Загружаем данные MOEX для {ticker}")
                data = get_moex_data(ticker, days=self.lookback_period + self.test_period + 10)
            else:
                print(f"Загружаем данные yfinance для {ticker}")
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if data is None or data.empty or len(data) < 20:
                print(f"Не удалось загрузить данные для {ticker}")
                return None

            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                print(f"Отсутствуют колонки {missing_columns} для {ticker}")
                return None

            if 'Date' not in data.columns:
                data = data.reset_index()
                if 'TRADEDATE' in data.columns:
                    data = data.rename(columns={'TRADEDATE': 'Date'})

            data = data.dropna()
            
            print(f"Загружено {len(data)} записей для {ticker}")
            return data
        
        except Exception as e:
            print(f"Ошибка загрузки данных для {ticker}: {e}")
            return None
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            if hasattr(self.model, '_add_technical_indicators'):
                return self.model._add_technical_indicators(data.copy())
            else:
                data = data.copy()

                data['SMA_5'] = data['Close'].rolling(window=5).mean()
                data['SMA_10'] = data['Close'].rolling(window=10).mean()
                data['SMA_20'] = data['Close'].rolling(window=20).mean()

                delta = data['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                data['RSI'] = 100 - (100 / (1 + rs))

                data['Volatility'] = data['Close'].rolling(window=20).std()
                data['Price_Change'] = data['Close'].pct_change()
                
                return data.dropna()
                
        except Exception as e:
            print(f"Ошибка при подготовке признаков: {e}")
            return data
    
    def get_news_for_period(self, ticker: str, date: datetime, is_moex: bool = False) -> List[str]:
        if not self.use_news:
            return []
            
        try:
            company_info = get_company_info_auto(ticker, is_moex)
            headlines = get_news_headlines(ticker, days=3, max_news=20, is_moex=is_moex)
            global_news = get_global_political_news(days=3, max_news=10, company_info=company_info)
            global_headlines = [item['title'] + '. ' + item['description'] for item in global_news]
            all_news = headlines + global_headlines
            
            return all_news
        except Exception as e:
            print(f"Ошибка получения новостей: {e}")
            return []
    
    def backtest_ticker(self, ticker: str, is_moex: bool = False) -> Dict:
        print(f"\n{'='*50}")
        print(f"Бэктестинг {ticker} {'(MOEX)' if is_moex else '(US)'}")
        print(f"{'='*50}")

        model_loaded = self.load_model_for_ticker(ticker)
        data = self.load_stock_data(ticker, is_moex=is_moex)
        if data is None:
            print(f"Не удалось загрузить данные для {ticker}")
            return None

        data_with_features = self.prepare_features(data)
        
        if len(data_with_features) < self.test_period + 10:
            print(f"Недостаточно данных для {ticker}: {len(data_with_features)}")
            return None
        test_data = data_with_features.iloc[-self.test_period:].copy()
        
        print(f"Тестовый период: {len(test_data)} дней")
        print(f"От {test_data.iloc[0]['Date']} до {test_data.iloc[-1]['Date']}")

        company_info = get_company_info_auto(ticker, is_moex)
        
        ticker_results = []

        for i in range(len(test_data) - self.prediction_horizon):
            try:
                current_date = test_data.iloc[i]['Date']
                features = test_data.iloc[i]

                news_text = None
                if self.use_news:
                    try:
                        news_list = self.get_news_for_period(ticker, current_date, is_moex)
                        news_text = ' '.join(news_list) if news_list else None
                    except:
                        news_text = None

                if self.model and hasattr(self.model, 'combine_predictions'):
                    pred_price, pred_direction, confidence = self.model.combine_predictions(
                        features, 
                        news_text=news_text,
                        is_moex=is_moex,
                        company_info=company_info
                    )
                else:
                    current_price = float(features['Close'])
                    pred_price = current_price * (1 + np.random.uniform(-0.02, 0.02))
                    pred_direction = 1 if pred_price > current_price else -1
                    confidence = 0.5

                actual_price = float(test_data.iloc[i + self.prediction_horizon]['Close'])
                current_price = float(features['Close'])

                price_change = actual_price - current_price
                if abs(price_change / current_price) < 0.005:
                    actual_direction = 0
                else:
                    actual_direction = 1 if price_change > 0 else -1

                price_error = abs(pred_price - actual_price)
                price_error_pct = (price_error / actual_price) * 100
                is_correct_direction = (pred_direction == actual_direction) or (actual_direction == 0)

                result = {
                    'ticker': ticker,
                    'date': current_date,
                    'actual_price': actual_price,
                    'predicted_price': pred_price,
                    'actual_direction': actual_direction,
                    'predicted_direction': pred_direction,
                    'confidence': confidence,
                    'price_error': price_error,
                    'price_error_pct': price_error_pct,
                    'is_correct_direction': is_correct_direction
                }
                
                ticker_results.append(result)

                for key, value in result.items():
                    self.results[key].append(value)
                
            except Exception as e:
                print(f"Ошибка прогноза для {ticker} на {current_date}: {e}")
                continue
        
        if not ticker_results:
            print(f"Нет валидных прогнозов для {ticker}")
            return None

        price_errors = [r['price_error'] for r in ticker_results]
        price_errors_pct = [r['price_error_pct'] for r in ticker_results]
        direction_correct = [r['is_correct_direction'] for r in ticker_results]
        
        mae = np.mean(price_errors)
        mape = np.mean(price_errors_pct)
        direction_accuracy = np.mean(direction_correct)
        rmse = np.sqrt(np.mean([e**2 for e in price_errors]))

        returns = []
        for r in ticker_results:
            if r['predicted_direction'] != 0:
                actual_return = (r['actual_price'] - r['actual_price'] / (1 + (r['actual_price'] - r['predicted_price']) / r['predicted_price'])) / r['actual_price']
                predicted_return = r['predicted_direction'] * abs(actual_return)
                returns.append(predicted_return)
        
        strategy_return = np.sum(returns) if returns else 0
        
        stats = {
            'ticker': ticker,
            'predictions_count': len(ticker_results),
            'mae': mae,
            'mape': mape,
            'rmse': rmse,
            'direction_accuracy': direction_accuracy,
            'strategy_return': strategy_return,
            'model_loaded': model_loaded
        }
        
        print(f"\nРезультаты для {ticker}:")
        print(f"  Количество прогнозов: {stats['predictions_count']}")
        print(f"  MAE: {stats['mae']:.2f}")
        print(f"  MAPE: {stats['mape']:.2f}%")
        print(f"  RMSE: {stats['rmse']:.2f}")
        print(f"  Точность направления: {stats['direction_accuracy']:.2%}")
        print(f"  Доходность стратегии: {stats['strategy_return']:.2%}")
        print(f"  Модель загружена: {'Да' if model_loaded else 'Нет (fallback)'}")
        
        return stats
    
    def backtest_multiple(self, tickers: List[str], is_moex_list: Optional[List[bool]] = None) -> pd.DataFrame:
        if is_moex_list is None:
            is_moex_list = [is_russian_ticker(ticker) for ticker in tickers]
            
        if len(tickers) != len(is_moex_list):
            raise ValueError("Длина списков tickers и is_moex_list должна совпадать")
            
        results_summary = []

        self.results = {
            'ticker': [],
            'date': [],
            'actual_price': [],
            'predicted_price': [],
            'actual_direction': [],
            'predicted_direction': [],
            'confidence': [],
            'price_error': [],
            'price_error_pct': [],
            'is_correct_direction': []
        }

        for ticker, is_moex in zip(tickers, is_moex_list):
            ticker_result = self.backtest_ticker(ticker, is_moex)
            if ticker_result:
                results_summary.append(ticker_result)
        
        if not results_summary:
            print("Нет валидных результатов для обобщения")
            return None
            
        results_df = pd.DataFrame(results_summary)

        overall_mae = results_df['mae'].mean()
        overall_mape = results_df['mape'].mean()
        overall_rmse = results_df['rmse'].mean()
        overall_dir_acc = results_df['direction_accuracy'].mean()
        overall_return = results_df['strategy_return'].mean()
        
        print(f"\n{'='*60}")
        print("ОБЩИЕ РЕЗУЛЬТАТЫ БЭКТЕСТИНГА")
        print(f"{'='*60}")
        print(f"Протестировано тикеров: {len(results_summary)}")
        print(f"Средняя MAE: {overall_mae:.2f}")
        print(f"Средняя MAPE: {overall_mape:.2f}%")
        print(f"Средняя RMSE: {overall_rmse:.2f}")
        print(f"Средняя точность направления: {overall_dir_acc:.2%}")
        print(f"Средняя доходность стратегии: {overall_return:.2%}")
        
        return results_df
    
    def save_results(self, output_dir: str = "backtest_results"):
        if not self.results['ticker']:
            print("Нет результатов для сохранения")
            return
            
        os.makedirs(output_dir, exist_ok=True)

        results_df = pd.DataFrame(self.results)
        results_path = os.path.join(output_dir, "detailed_results.csv")
        results_df.to_csv(results_path, index=False)
        print(f"Детальные результаты сохранены в {results_path}")

        summary_data = []
        for ticker in results_df['ticker'].unique():
            ticker_data = results_df[results_df['ticker'] == ticker]
            summary_data.append({
                'ticker': ticker,
                'predictions': len(ticker_data),
                'mae': ticker_data['price_error'].mean(),
                'mape': ticker_data['price_error_pct'].mean(),
                'direction_accuracy': ticker_data['is_correct_direction'].mean(),
                'avg_confidence': ticker_data['confidence'].mean()
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(output_dir, "summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"Сводка сохранена в {summary_path}")
    
    def plot_results(self, output_dir: str = "backtest_results"):
        if not self.results['ticker']:
            print("Нет результатов для построения графиков")
            return
            
        os.makedirs(output_dir, exist_ok=True)
        results_df = pd.DataFrame(self.results)

        try:
            results_df['date'] = pd.to_datetime(results_df['date'])
        except Exception as e:
            print(f"Ошибка преобразования дат: {e}")
            # Если не удается преобразовать даты, используем индексы
            results_df['date_index'] = range(len(results_df))

        plt.style.use('default')
        sns.set_palette("husl")
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            for ticker in results_df['ticker'].unique():
                ticker_data = results_df[results_df['ticker'] == ticker].copy()

                if 'date' in ticker_data.columns and pd.api.types.is_datetime64_any_dtype(ticker_data['date']):
                    ticker_data = ticker_data.sort_values('date')
                    x_axis = ticker_data['date']
                    x_label = 'Дата'
                else:
                    if 'date_index' not in ticker_data.columns:
                        ticker_data['date_index'] = range(len(ticker_data))
                    ticker_data = ticker_data.sort_values('date_index')
                    x_axis = ticker_data['date_index']
                    x_label = 'Индекс'
                
                axes[0, 0].plot(x_axis, ticker_data['actual_price'], 
                               label=f'{ticker} Факт', linewidth=2)
                axes[0, 0].plot(x_axis, ticker_data['predicted_price'], 
                               label=f'{ticker} Прогноз', linestyle='--', alpha=0.7)
            
            axes[0, 0].set_title('Фактические vs Прогнозируемые цены')
            axes[0, 0].set_xlabel(x_label)
            axes[0, 0].set_ylabel('Цена')
            axes[0, 0].legend()
            if x_label == 'Дата':
                axes[0, 0].tick_params(axis='x', rotation=45)

            axes[0, 1].hist(results_df['price_error_pct'], bins=30, alpha=0.7, edgecolor='black')
            axes[0, 1].set_title('Распределение ошибок прогноза (%)')
            axes[0, 1].set_xlabel('Ошибка (%)')
            axes[0, 1].set_ylabel('Частота')
            mean_error = results_df['price_error_pct'].mean()
            axes[0, 1].axvline(mean_error, color='red', 
                              linestyle='--', label=f'Среднее: {mean_error:.2f}%')
            axes[0, 1].legend()

            direction_by_ticker = results_df.groupby('ticker')['is_correct_direction'].mean()
            bars = axes[1, 0].bar(direction_by_ticker.index, direction_by_ticker.values)
            axes[1, 0].set_title('Точность прогноза направления по тикерам')
            axes[1, 0].set_xlabel('Тикер')
            axes[1, 0].set_ylabel('Точность')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].axhline(y=0.5, color='red', linestyle='--', label='Случайное угадывание')
            axes[1, 0].legend()

            for bar, value in zip(bars, direction_by_ticker.values):
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{value:.1%}', ha='center', va='bottom')

            try:
                if 'date' in results_df.columns and pd.api.types.is_datetime64_any_dtype(results_df['date']):
                    daily_accuracy = results_df.groupby('date')['is_correct_direction'].mean()
                    if len(daily_accuracy) > 5:
                        daily_accuracy = daily_accuracy.rolling(window=5, min_periods=1).mean()
                    
                    axes[1, 1].plot(daily_accuracy.index, daily_accuracy.values, marker='o', markersize=4)
                    axes[1, 1].set_title('Точность направления во времени')
                    axes[1, 1].set_xlabel('Дата')
                    axes[1, 1].tick_params(axis='x', rotation=45)
                else:
                    if 'date_index' not in results_df.columns:
                        results_df['date_index'] = range(len(results_df))
                    accuracy_by_index = results_df.groupby('date_index')['is_correct_direction'].mean()
                    axes[1, 1].plot(accuracy_by_index.index, accuracy_by_index.values, marker='o', markersize=4)
                    axes[1, 1].set_title('Точность направления по порядку')
                    axes[1, 1].set_xlabel('Порядковый номер')
                
                axes[1, 1].set_ylabel('Точность')
                axes[1, 1].axhline(y=0.5, color='red', linestyle='--', label='Случайное угадывание')
                axes[1, 1].set_ylim(0, 1)
                axes[1, 1].legend()
                
            except Exception as e:
                print(f"Ошибка при построении графика точности во времени: {e}")
                overall_accuracy = results_df['is_correct_direction'].mean()
                axes[1, 1].bar(['Общая точность'], [overall_accuracy])
                axes[1, 1].set_title('Общая точность направления')
                axes[1, 1].set_ylabel('Точность')
                axes[1, 1].axhline(y=0.5, color='red', linestyle='--', label='Случайное угадывание')
                axes[1, 1].legend()
            
            plt.tight_layout()
            plot_path = os.path.join(output_dir, "backtest_analysis.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"График сохранен в {plot_path}")

            try:
                plt.show()
            except:
                print("Не удалось отобразить график (возможно, нет GUI)")
                
        except Exception as e:
            print(f"Ошибка при построении графиков: {e}")
            import traceback
            traceback.print_exc()

            try:
                fig, ax = plt.subplots(1, 1, figsize=(10, 6))

                summary_data = []
                for ticker in results_df['ticker'].unique():
                    ticker_data = results_df[results_df['ticker'] == ticker]
                    summary_data.append({
                        'ticker': ticker,
                        'accuracy': ticker_data['is_correct_direction'].mean(),
                        'mape': ticker_data['price_error_pct'].mean()
                    })
                
                summary_df = pd.DataFrame(summary_data)
                
                x = range(len(summary_df))
                ax.bar([i - 0.2 for i in x], summary_df['accuracy'], 0.4, label='Точность направления', alpha=0.7)
                ax2 = ax.twinx()
                ax2.bar([i + 0.2 for i in x], summary_df['mape'], 0.4, label='MAPE (%)', alpha=0.7, color='orange')
                
                ax.set_xlabel('Тикеры')
                ax.set_ylabel('Точность направления')
                ax2.set_ylabel('MAPE (%)')
                ax.set_title('Сводка результатов бэктестинга')
                ax.set_xticks(x)
                ax.set_xticklabels(summary_df['ticker'])
                
                ax.legend(loc='upper left')
                ax2.legend(loc='upper right')
                
                plt.tight_layout()
                simple_plot_path = os.path.join(output_dir, "simple_summary.png")
                plt.savefig(simple_plot_path, dpi=300, bbox_inches='tight')
                print(f"Упрощенный график сохранен в {simple_plot_path}")
                
                try:
                    plt.show()
                except:
                    print("Не удалось отобразить упрощенный график")
                    
            except Exception as e2:
                print(f"Не удалось создать даже упрощенный график: {e2}")
        
        finally:
            plt.close('all')


def main():
    parser = argparse.ArgumentParser(description="Бэктестинг модели прогнозирования акций")
    parser.add_argument("--model_path", type=str, default="models",
                        help="Путь к файлам модели")
    parser.add_argument("--tickers", type=str, default="AAPL,SBER,GAZP", 
                        help="Список тикеров через запятую")
    parser.add_argument("--lookback_period", type=int, default=60, 
                        help="Количество дней для обучающих данных")
    parser.add_argument("--test_period", type=int, default=30, 
                        help="Количество дней для тестирования")
    parser.add_argument("--prediction_horizon", type=int, default=1, 
                        help="Горизонт прогнозирования (дни)")
    parser.add_argument("--use_news", action="store_true", 
                        help="Включить новостные данные")
    parser.add_argument("--use_macro", action="store_true", 
                        help="Включить макроэкономические данные")
    parser.add_argument("--output_dir", type=str, default="backtest_results", 
                        help="Директория для сохранения результатов")
    parser.add_argument("--plot", action="store_true", default=True,
                        help="Построить графики результатов")
    
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(',')]
    
    print("Запуск бэктестинга с параметрами:")
    print(f"  Тикеры: {tickers}")
    print(f"  Путь к модели: {args.model_path}")
    print(f"  Период обучения: {args.lookback_period} дней")
    print(f"  Период тестирования: {args.test_period} дней")
    print(f"  Горизонт прогноза: {args.prediction_horizon} день(дня)")
    print(f"  Использовать новости: {args.use_news}")
    print(f"  Использовать макро данные: {args.use_macro}")

    backtester = ModelBacktester(
        model_path=args.model_path,
        lookback_period=args.lookback_period,
        test_period=args.test_period,
        prediction_horizon=args.prediction_horizon,
        use_news=args.use_news,
        use_macro=args.use_macro
    )

    results = backtester.backtest_multiple(tickers)

    if results is not None:
        backtester.save_results(args.output_dir)

        if args.plot:
            backtester.plot_results(args.output_dir)
    
    return results


if __name__ == "__main__":
    main() 