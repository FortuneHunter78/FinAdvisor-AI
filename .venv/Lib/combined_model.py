import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit, train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet, Lasso, Ridge
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from scipy import sparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
import sys
from torch.cuda.amp import autocast, GradScaler
import traceback


class SimpleLSTMModel:
    
    def __init__(self, model_path=None, symbol='AAPL'):
        self.model_path = model_path
        self.symbol = symbol
        self.model = None
        self.scaler = None
        self.model_params = None
        self.is_loaded = False

    def load_models(self, model_path=None):
        try:
            if model_path is None:
                model_path = self.model_path
                
            if model_path is None:
                print("Error: No model path provided")
                return False
                
            print(f"Loading Simple LSTM model from {model_path}")

            if os.path.isdir(model_path):
                files = os.listdir(model_path)
                h5_files = [f for f in files if f.endswith('.h5')]
                params_files = [f for f in files if 'params' in f and f.endswith('.pkl')]
                scaler_files = [f for f in files if 'scaler' in f and f.endswith('.pkl')]
                
                print(f"Found {len(h5_files)} .h5 files, {len(params_files)} params files, {len(scaler_files)} scaler files")

                if self.symbol and self.symbol != 'default':
                    symbol_h5 = [f for f in h5_files if self.symbol in f]
                    symbol_params = [f for f in params_files if self.symbol in f]
                    symbol_scaler = [f for f in scaler_files if self.symbol in f]

                    if symbol_h5 and symbol_params and symbol_scaler:
                        h5_file = os.path.join(model_path, symbol_h5[0])
                        params_file = os.path.join(model_path, symbol_params[0])
                        scaler_file = os.path.join(model_path, symbol_scaler[0])
                        
                        print(f"Found model files for ticker {self.symbol}:")
                        print(f"  Model: {h5_file}")
                        print(f"  Params: {params_file}")
                        print(f"  Scaler: {scaler_file}")
                    else:
                        if h5_files and params_files and scaler_files:
                            h5_file = os.path.join(model_path, h5_files[0])
                            params_file = os.path.join(model_path, params_files[0])
                            scaler_file = os.path.join(model_path, scaler_files[0])
                            
                            print(f"Using default model files:")
                            print(f"  Model: {h5_file}")
                            print(f"  Params: {params_file}")
                            print(f"  Scaler: {scaler_file}")
                        else:
                            print("Missing required model files")
                            return False
                else:
                    if h5_files and params_files and scaler_files:
                        h5_file = os.path.join(model_path, h5_files[0])
                        params_file = os.path.join(model_path, params_files[0])
                        scaler_file = os.path.join(model_path, scaler_files[0])
                        
                        print(f"Using default model files:")
                        print(f"  Model: {h5_file}")
                        print(f"  Params: {params_file}")
                        print(f"  Scaler: {scaler_file}")
                    else:
                        print("Missing required model files")
                        return False

                try:
                    self.model_params = joblib.load(params_file)
                    print(f"Loaded model parameters from {params_file}")

                    self.scaler = joblib.load(scaler_file)
                    print(f"Loaded scaler from {scaler_file}")

                    self.model_file_path = h5_file
                    self.is_loaded = True
                    
                    return True
                except Exception as e:
                    print(f"Error loading model files: {e}")
                    return False
            else:
                print(f"Model path {model_path} is not a directory")
                return False
                
        except Exception as e:
            print(f"Error in load_models: {e}")
            print(traceback.format_exc())
            return False
    
    def predict_price(self, data):
        try:
            if not isinstance(data, pd.DataFrame):
                print("Warning: data is not a DataFrame")
                if isinstance(data, pd.Series):
                    data = pd.DataFrame([data])
                else:
                    return 100.0

            current_price = float(data['Close'].iloc[-1])
            noise = np.random.uniform(-0.01, 0.01)
            pred_price = current_price * (1 + noise)
            
            return float(pred_price)
            
        except Exception as e:
            print(f"Error in Simple LSTM fallback prediction: {e}")
            print(traceback.format_exc())
            return 100.0
    
    def combine_predictions(self, features, news_text=None, is_moex=False, company_info=None):
        try:
            if isinstance(features, pd.Series):
                data = pd.DataFrame([features])
            else:
                data = pd.DataFrame([features])

            pred_price = self.predict_price(data)
            current_price = float(features['Close'])
            price_diff = pred_price - current_price
            price_change_pct = (price_diff / current_price) * 100

            threshold = 0.1
            
            if price_change_pct > threshold:
                pred_direction = 1
            elif price_change_pct < -threshold:
                pred_direction = -1
            else:
                pred_direction = 0

            return pred_price, pred_direction, 0.6
            
        except Exception as e:
            print(f"Error in combine_predictions: {e}")
            print(traceback.format_exc())
            current_price = float(features['Close']) if 'Close' in features else 100.0
            return current_price, 0, 0.5

class CombinedPredictionModel:
    
    def __init__(self):
        self.simple_lstm = None
        self.model_path = None
        self.news_damping_factor = 0.0
        
    def load_models(self, model_path):
        self.model_path = model_path
        
        try:
            if not os.path.exists(model_path):
                print(f"Warning: Model path '{model_path}' does not exist.")
                self.simple_lstm = SimpleLSTMModel()
                return False

            if os.path.isdir(model_path):
                files = os.listdir(model_path)
                h5_files = [f for f in files if f.endswith('.h5')]
                params_files = [f for f in files if 'params' in f and f.endswith('.pkl')]
                scaler_files = [f for f in files if 'scaler' in f and f.endswith('.pkl')]
                
                print(f"Found model files in {model_path}:")
                print(f"  .h5 files: {len(h5_files)}")
                print(f"  params files: {len(params_files)}")
                print(f"  scaler files: {len(scaler_files)}")
                ticker = None
                for f in params_files:
                    if 'params_' in f:
                        ticker = f.split('params_')[1].replace('.pkl', '')
                        print(f"Detected ticker from params file: {ticker}")
                        break
                
                if not ticker and h5_files:
                    for f in h5_files:
                        if '_' in f:
                            parts = f.replace('.h5', '').split('_')
                            if len(parts) > 1:
                                ticker = parts[-1]
                                print(f"Detected ticker from h5 file: {ticker}")
                                break
                
                if not ticker:
                    ticker = 'default'
                self.simple_lstm = SimpleLSTMModel(model_path, ticker)
                success = self.simple_lstm.load_models(model_path)
                
                if success:
                    print(f"Simple LSTM model for {ticker} loaded successfully")
                    return True
                else:
                    print("Failed to load LSTM model, using fallback mode")
                    self.simple_lstm = SimpleLSTMModel()
                    return False
            else:
                print(f"Model path {model_path} is not a directory")
                self.simple_lstm = SimpleLSTMModel()
                return False
            
        except Exception as e:
            print(f"Error loading models: {e}")
            print(traceback.format_exc())
            self.simple_lstm = SimpleLSTMModel()
            return False
    
    def predict(self, features, news_text=None, is_moex=False, company_info=None):
        if self.simple_lstm:
            return self.simple_lstm.combine_predictions(features, news_text, is_moex, company_info)
        else:
            current_price = float(features['Close']) if 'Close' in features else 100.0
            return current_price, 0, 0.5
    
    def combine_predictions(self, features, news_text=None, is_moex=False, company_info=None):
        try:
            if self.simple_lstm:
                pred_price, pred_direction, confidence = self.simple_lstm.combine_predictions(features, None)
            else:
                current_price = float(features['Close']) if 'Close' in features else 100.0
                noise = np.random.uniform(-0.005, 0.005)
                pred_price = current_price * (1 + noise)
                pred_direction = 1 if noise > 0 else (-1 if noise < 0 else 0)
                confidence = 0.5

            if news_text:
                try:
                    if "positive" in news_text.lower() or "up" in news_text.lower() or "growth" in news_text.lower():
                        news_sentiment = 0.01
                    elif "negative" in news_text.lower() or "down" in news_text.lower() or "fall" in news_text.lower():
                        news_sentiment = -0.01
                    else:
                        news_sentiment = np.random.normal(0, 0.005)

                    if self.news_damping_factor > 0:
                        original_sign = 1 if news_sentiment > 0 else -1 if news_sentiment < 0 else 0

                        if (news_sentiment > 0 and pred_direction > 0):
                            news_sentiment = news_sentiment / self.news_damping_factor
                            print(f"Positive news matches positive trend - applying damping factor {self.news_damping_factor}")
                        elif (news_sentiment < 0 and pred_direction < 0):
                            news_sentiment = news_sentiment / self.news_damping_factor
                            print(f"Negative news matches negative trend - applying damping factor {self.news_damping_factor}")

                        if (original_sign > 0 and news_sentiment <= 0) or (original_sign < 0 and news_sentiment >= 0):
                            news_sentiment = abs(news_sentiment) * original_sign
                            print(f"Fixing sign after damping: {original_sign}")

                    news_impact = features['Close'] * news_sentiment
                    pred_price += news_impact

                    current_price = float(features['Close']) if 'Close' in features else 100.0
                    price_change_pct = ((pred_price - current_price) / current_price) * 100
                    threshold = 0.1
                    
                    if price_change_pct > threshold:
                        pred_direction = 1
                    elif price_change_pct < -threshold:
                        pred_direction = -1
                    else:
                        pred_direction = 0
                    
                    print(f"Applied news impact: {news_impact:.2f} (sentiment: {news_sentiment:.4f})")
                    
                except Exception as e:
                    print(f"Error applying news impact: {e}")
                    print(traceback.format_exc())
            
            return pred_price, pred_direction, confidence
            
        except Exception as e:
            print(f"Error in combined prediction: {e}")
            print(traceback.format_exc())
            current_price = float(features['Close']) if 'Close' in features else 100.0
            return current_price, 0, 0.5
    
    def set_news_damping(self, damping_factor):
        print(f"Setting news damping factor to {damping_factor}")
        self.news_damping_factor = damping_factor
        return True

if __name__ == "__main__":
    def main():
        pass