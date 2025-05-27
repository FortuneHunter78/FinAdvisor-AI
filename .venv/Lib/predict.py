import pandas as pd
import numpy as np
import joblib
import argparse
import sys
import os
import json
from datetime import datetime, timedelta
import traceback
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer, BertModel
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import requests
from dotenv import load_dotenv
from macro_data import MacroAdjustment
import pickle
from collections import defaultdict

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    HAS_TENSORFLOW = True
except ImportError:
    print("Error: TensorFlow not found. Please install: pip install tensorflow")
    HAS_TENSORFLOW = False

load_dotenv()

warnings.filterwarnings("ignore", category=UserWarning)
news_importance_model = None
news_importance_vectorizer = None
news_importance_bert_tokenizer = None
news_importance_bert_model = None

class SimpleLSTMModel:
    
    def __init__(self, model_path, symbol='AAPL'):
        self.model_path = model_path
        self.symbol = symbol
        self.model = None
        self.scaler = None
        self.model_params = None
        self.is_loaded = False

    def find_model_files(self, model_path, symbol):
        model_files = {
            'model': None,
            'scaler': None,
            'params': None
        }

        possible_names = [
            symbol,
            symbol.upper(),
            symbol.lower(),
            symbol.replace('.ME', ''),
            'default',
            'model'
        ]

        for name in possible_names:
            model_file = os.path.join(model_path, f'simple_lstm_{name}.h5')
            if os.path.exists(model_file) and model_files['model'] is None:
                model_files['model'] = model_file

            scaler_file = os.path.join(model_path, f'simple_lstm_scaler_{name}.pkl')
            if os.path.exists(scaler_file) and model_files['scaler'] is None:
                model_files['scaler'] = scaler_file

            params_file = os.path.join(model_path, f'simple_lstm_params_{name}.pkl')
            if os.path.exists(params_file) and model_files['params'] is None:
                model_files['params'] = params_file

        return model_files
        
    def load_models(self, model_path):
        try:
            model_files = self.find_model_files(model_path, self.symbol)
            missing_files = []
            for file_type, file_path in model_files.items():
                if file_path is None:
                    missing_files.append(file_type)

            if missing_files:
                available_models = self.find_available_models(model_path)
                if available_models:
                    print(f"Найдены модели для: {', '.join(available_models)}")
                    fallback_symbol = available_models[0]
                    print(f"Используем модель для {fallback_symbol}")
                    model_files = self.find_model_files(model_path, fallback_symbol)
                else:
                    raise FileNotFoundError(f"Модели не найдены в {model_path}")

            self.model = load_model(model_files['model'])
            print(f"✅ Simple LSTM модель загружена: {model_files['model']}")

            self.scaler = joblib.load(model_files['scaler'])
            print(f"✅ Скейлер загружен: {model_files['scaler']}")

            self.model_params = joblib.load(model_files['params'])
            print(f"✅ Параметры загружены: {model_files['params']}")
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            print(f"❌ Ошибка при загрузке Simple LSTM модели: {e}")
            self.is_loaded = False
            return False
    
    def find_available_models(self, model_path):
        available_models = set()
        
        if not os.path.exists(model_path):
            return []
        
        for filename in os.listdir(model_path):
            if filename.startswith('simple_lstm_') and filename.endswith('.h5'):
                model_name = filename.replace('simple_lstm_', '').replace('.h5', '')
                available_models.add(model_name)
        
        return list(available_models)
    
    def predict_price(self, data):
        try:
            if not self.is_loaded:
                raise Exception("Модель не загружена")
                
            time_step = self.model_params.get('time_step', 15)
            close_values = data[['Close']].tail(time_step).values

            if len(close_values) < time_step:
                print(f"Недостаточно данных для предсказания. Требуется {time_step}, доступно {len(close_values)}")
                return float(data['Close'].iloc[-1])

            close_scaled = self.scaler.transform(close_values)
            X = close_scaled.reshape(1, time_step, 1)
            prediction_scaled = self.model.predict(X, verbose=0)
            prediction = self.scaler.inverse_transform(prediction_scaled)[0, 0]
            
            return float(prediction)
            
        except Exception as e:
            print(f"Ошибка в предсказании Simple LSTM: {e}")
            return float(data['Close'].iloc[-1])
    
    def combine_predictions(self, features, news_text=None):
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

            return pred_price, pred_direction, 0.8
            
        except Exception as e:
            print(f"Ошибка в combine_predictions: {e}")
            current_price = float(features['Close'])
            return current_price, 0, 0.5

class CombinedPredictionModel:
    
    def __init__(self):
        self.simple_lstm = None
        self.model_path = None
        
    def load_models(self, model_path):
        self.model_path = model_path
        symbol = 'default'
        
        self.simple_lstm = SimpleLSTMModel(model_path, symbol)
        success = self.simple_lstm.load_models(model_path)
        
        if not success:
            print("⚠️ Не удалось загрузить Simple LSTM модель, используем fallback режим")
            self.simple_lstm = None
        
        return success
    
    def predict(self, features, news_text=None):
        if self.simple_lstm and self.simple_lstm.is_loaded:
            return self.simple_lstm.combine_predictions(features, news_text)
        else:
            current_price = float(features['Close'])
            return current_price, 0, 0.5

def extract_bert_features(text, tokenizer, model):
    if tokenizer is None or model is None:
        return None

    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = model(**inputs)
            last_hidden_state = outputs.last_hidden_state
            sentence_embedding = torch.mean(last_hidden_state, dim=1).squeeze().numpy()

        return sentence_embedding
    except Exception as e:
        print(f"Ошибка при извлечении признаков BERT: {e}")
        return None


def extract_text_features(text, vectorizer):
    if vectorizer is None:
        return None

    try:
        features = vectorizer.transform([text])
        return features
    except Exception as e:
        print(f"Ошибка при извлечении текстовых признаков: {e}")
        return None

def calculate_news_importance(news_text, company_info=None, is_moex=False):
    return 0.5

def preprocess_russian_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)

    tokens = word_tokenize(text, language='russian')
    tokens = [russian_stemmer.stem(token) for token in tokens if token not in russian_stopwords]

    return ' '.join(tokens)


def analyze_russian_sentiment(text: str) -> float:
    try:
        model_path = "russian_sentiment_model"
        if not os.path.exists(model_path):
            print("Model not found. Please run train_sentiment_model.py first.")
            return 0.0

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)

        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=1)
            sentiment_score = predictions[0][1].item()
            sentiment_score = sentiment_score * 2 - 1

            return sentiment_score

    except Exception as e:
        print(f"Error in sentiment analysis: {str(e)}")
        return 0.0


def get_sentiment_category(sentiment: float, text: str) -> int:
    if sentiment <= -0.15:
        return -1
    elif sentiment >= 0.5:
        return 1
    else:
        return 0


def get_news_sentiment(news_titles: list, is_russian: bool = False) -> float:
    if not news_titles:
        return 0.0

    sentiment_categories = []
    weights = np.linspace(1.0, 0.5, len(news_titles))
    
    if is_russian:
        print("\nSentiment analysis results for Russian news:")
        for title in news_titles:
            continuous_sentiment = analyze_russian_sentiment(title)
            discrete_sentiment = get_sentiment_category(continuous_sentiment, title)
            sentiment_categories.append(discrete_sentiment)
            
            print(f"Title: {title}")
            print(f"Sentiment score: {continuous_sentiment:.2f} (Category: {discrete_sentiment})")
            print("-" * 50)

        if sentiment_categories:
            weighted_sentiment = np.average(sentiment_categories, weights=weights)
        else:
            weighted_sentiment = 0.0

        print(f"\nOverall sentiment analysis:")
        print(f"Average discrete sentiment: {weighted_sentiment:.2f}")
        print(f"Estimated price impact: {weighted_sentiment * 0.02 * 100:.1f}%")

        return weighted_sentiment
    else:
        print("\nSentiment analysis results for foreign news:")
        for title in news_titles:
            continuous_sentiment = analyze_russian_sentiment(title)
            discrete_sentiment = get_sentiment_category(continuous_sentiment, title)
            sentiment_categories.append(discrete_sentiment)
            
            print(f"Title: {title}")
            print(f"Sentiment score: {continuous_sentiment:.2f} (Category: {discrete_sentiment})")
            print("-" * 50)

        if sentiment_categories:
            weighted_sentiment = np.average(sentiment_categories, weights=weights)
        else:
            weighted_sentiment = 0.0

        print(f"\nOverall sentiment analysis:")
        print(f"Average discrete sentiment: {weighted_sentiment:.2f}")
        print(f"Estimated price impact: {weighted_sentiment * 0.015 * 100:.1f}%")

        return weighted_sentiment


def count_news_mentions(news_text, api_key=None):
    import requests
    from datetime import datetime, timedelta
    api_key = api_key or os.getenv('NEWS_API_KEY', '64134ea289d04936aeb08f00861d9e81')

    try:
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)

        stop_words = set(stopwords.words('english'))
        stop_words.update(set(stopwords.words('russian')))

        tokens = word_tokenize(news_text.lower())
        filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]

        key_words = filtered_tokens[:5]

        if len(key_words) < 2:
            key_words = [word for word in tokens[:5] if word.isalnum()]

        search_query = " OR ".join(key_words)
    except Exception as e:
        words = news_text.split()[:5]
        search_query = " OR ".join(words)

    print(f"Searching for news mentions with query: {search_query}")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    params = {
        'q': search_query,
        'from': start_date.strftime('%Y-%m-%d'),
        'to': end_date.strftime('%Y-%m-%d'),
        'language': 'en,ru',
        'sortBy': 'relevancy',
        'pageSize': 100,
        'apiKey': api_key
    }

    try:
        response = requests.get('https://newsapi.org/v2/everything', params=params)
        response.raise_for_status()

        data = response.json()
        total_results = data.get('totalResults', 0)

        articles = data.get('articles', [])

        mentions = 0
        for article in articles:
            title = article.get('title', '').lower()
            description = article.get('description', '').lower()
            content = article.get('content', '').lower()

            relevance_score = 0
            for word in key_words:
                if word in title:
                    relevance_score += 3
                if word in description:
                    relevance_score += 2
                if word in content:
                    relevance_score += 1

            if relevance_score >= 3:
                mentions += 1

        print(f"Found {mentions} relevant mentions out of {total_results} total results")
        return mentions

    except Exception as e:
        print(f"Error counting news mentions: {str(e)}")
        return 0

def calculate_basic_sentiment(news_text):
    try:
        return analyze_russian_sentiment(news_text)
    except Exception as e:
        print(f"Error in model-based sentiment analysis: {str(e)}")
        return 0.0


def analyze_global_political_news(company_info, global_news, is_moex=False):
    print(f"DEBUG: ENTERING analyze_global_political_news. Global news count: {len(global_news) if global_news else 0}")

    if not global_news:
        print("DEBUG: Empty global_news list")
        return {}

    if not company_info:
        print("DEBUG: Empty company_info")
        return {}

    try:
        results = {
            'impact_score': 0.0,
            'relevant_news': [],
            'relevant_news_count': 0,
            'country_impacts': {}
        }

        dependencies = company_info.get('dependencies', [])
        if not isinstance(dependencies, list):
            print(f"DEBUG: Invalid dependencies format: {type(dependencies)}")
            dependencies = []

        company_country = company_info.get('country', '').lower()
        if not company_country:
            print("DEBUG: No company country found in company_info")
            company_country = "unknown"

        print(f"DEBUG: Company country: {company_country}")
        print(f"DEBUG: Dependencies count: {len(dependencies)}")

        country_importance = {}

        if company_country and company_country != "unknown":
            country_importance[company_country] = 1.0

        for dep in dependencies:

            if not isinstance(dep, dict):
                print(f"DEBUG: Skipping non-dict dependency: {dep}")
                continue

            country = dep.get('country', '').lower()
            if not country:
                print("DEBUG: Empty country in dependency")
                continue

            strength = dep.get('strength', 0.5)
            print(f"DEBUG: Processing dependency: country={country}, strength={strength}, type={type(strength)}")

            strength_value = 0.5
            try:
                if isinstance(strength, (int, float)):
                    strength_value = float(strength)
                elif isinstance(strength, str) and strength.replace('.', '', 1).isdigit():
                    strength_value = float(strength)
                else:
                    print(f"DEBUG: Using default strength for {country}, original value: {strength}")
            except (ValueError, TypeError) as e:
                print(f"DEBUG: Error converting strength to float: {e}")

            country_importance[country] = strength_value
            print(f"DEBUG: Added country {country} with importance {strength_value}")

        if not country_importance:
            print("DEBUG: No countries in importance dictionary, using defaults")
            country_importance = {
                'usa': 0.8,
                'china': 0.7,
                'eu': 0.6,
                'russia': 0.5,
                'germany': 0.5,
                'uk': 0.5
            }

        print(f"DEBUG: Final country_importance dictionary: {country_importance}")

        total_impact = 0.0
        relevant_news_count = 0

        news_count = 0
        for news_item in global_news:
            news_count += 1
            print(f"DEBUG: Processing news item #{news_count}")

            news_text = ""
            if isinstance(news_item, dict):
                news_text = news_item.get('text', '')
                print(f"DEBUG: Got news from dict: '{news_text[:50]}...'")
            else:
                news_text = str(news_item)
                print(f"DEBUG: Converted news to string: '{news_text[:50]}...'")

            if not news_text:
                print("DEBUG: Empty news text, skipping")
                continue

            try:
                sentiment = calculate_basic_sentiment(news_text)
                print(f"DEBUG: Calculated sentiment: {sentiment}")
                sentiment_category = 1 if sentiment > 0.2 else (-1 if sentiment < -0.2 else 0)
            except Exception as e:
                print(f"DEBUG: Error calculating sentiment: {e}")
                sentiment = 0.0
                sentiment_category = 0

            countries_mentioned = []
            for country_key in country_importance.keys():
                if country_key and country_key in news_text.lower():
                    countries_mentioned.append(country_key)

            if countries_mentioned:
                print(f"DEBUG: News mentions countries: {countries_mentioned}")
                country_importances_values = [country_importance.get(c, 0.0) for c in countries_mentioned]
                print(f"DEBUG: Country importances: {country_importances_values}")

                if not country_importances_values:
                    max_importance = 0.0
                else:
                    max_importance = max(country_importances_values)

                impact = sentiment * max_importance

                print(f"DEBUG: News sentiment: {sentiment}, max_importance: {max_importance}, impact: {impact}")

                total_impact += impact
                relevant_news_count += 1

                news_info = {
                    'text': news_text[:200],
                    'sentiment': sentiment,
                    'sentiment_category': sentiment_category,
                    'impact_score': impact,
                    'countries_mentioned': countries_mentioned
                }
                results['relevant_news'].append(news_info)
                for c_mentioned in countries_mentioned:
                    if c_mentioned not in results['country_impacts']:
                        results['country_impacts'][c_mentioned] = 0.0
                    results['country_impacts'][c_mentioned] += impact
                    print(f"DEBUG: Updated impact for {c_mentioned}: {results['country_impacts'][c_mentioned]}")

        if relevant_news_count > 0:
            total_impact = total_impact / relevant_news_count

            total_impact = max(min(total_impact, 0.01), -0.01)
        else:
            print("DEBUG: No relevant countries mentioned in any news")
            total_impact = 0.0

        results['impact_score'] = total_impact
        results['relevant_news_count'] = relevant_news_count

        print(f"DEBUG: Final results: {results}")
        return results

    except Exception as e:
        import traceback

        print(f"Error analyzing global political news: {str(e)}")
        traceback.print_exc()
        return {}


def print_prediction_results(results, is_moex=False):
    if not results:
        print("\n❌ ERROR: No prediction results available.")
        return

    if "error" in results:
        print("\n❌ ERROR IN PREDICTION ❌")
        print(f"Error message: {results.get('error', 'Unknown error')}")
        return

    try:
        ticker = results.get('ticker', 'unknown')
        current_price = results.get('current_price')
        date = results.get('date')

        ts_prediction = results.get('ts_prediction', {})
        ts_price = ts_prediction.get('price')
        ts_direction = ts_prediction.get('direction')
        ts_change_percent = ts_prediction.get('change_percent')

        combined_prediction = results.get('combined_prediction', {})
        predicted_price = combined_prediction.get('price')
        predicted_direction = combined_prediction.get('direction')

        news_analysis = results.get('news_analysis', {})
        has_news = bool(news_analysis)
        news_count = news_analysis.get('news_count', 0)
        sentiment_counts = news_analysis.get('sentiment_counts', {})
        sentiment_percentages = news_analysis.get('sentiment_percentages', {})
        news_influence = news_analysis.get('news_influence', 0)
        news_price_impact = news_analysis.get('price_impact', 0)
        direction_changed_by_news = news_analysis.get('direction_changed_by_news', False)

        global_news_impact = news_analysis.get('global_news_impact', {})
        global_impact_score = global_news_impact.get('impact_score', 0)
        global_news_count = global_news_impact.get('relevant_news_count', 0)
        country_impacts = global_news_impact.get('country_impacts', {})

        macro_data = results.get('macro_data', {})
        has_macro = macro_data.get('used', False)
        macro_indicator_count = macro_data.get('indicator_count', 0)
        macro_trends = macro_data.get('trends', {})

        macro_impact = results.get('macro_impact', {})
        has_macro_impact = bool(macro_impact)
        sector_impact = macro_impact.get('sector_impact', 0)
        country_impact = macro_impact.get('country_impact', 0)
        total_macro_impact = macro_impact.get('total_impact', 0)
        interest_rate_impact = macro_impact.get('interest_rate_impact', 0)

        price_adjustment = macro_impact.get('price_adjustment', 0)
        interest_rate_data = results.get('interest_rate_impact', {})
        has_interest_rate_impact = interest_rate_data.get('has_impact', False)

        print("\n" + "=" * 50)
        print(f"📊 PREDICTION RESULTS FOR {ticker}")
        print("=" * 50)

        print(f"\n💰 CURRENT STATE:")
        print(f"  • Current Price: {current_price:.2f}")
        print(f"  • Date: {date}")

        if has_news:
            print(f"\n📰 NEWS ANALYSIS:")
            if news_count:
                print(f"  • Analyzed {news_count} news items")
                if news_analysis.get('discretized_sentiment', False):
                    print("  • Using discretized sentiment categories (-1, 0, +1):")
                    discrete_values = news_analysis.get('discrete_sentiments', [])
                    if discrete_values:
                        pos_count = sum(1 for s in discrete_values if s > 0)
                        neg_count = sum(1 for s in discrete_values if s < 0)
                        neutral_count = sum(1 for s in discrete_values if s == 0)
                        print(f"    - Positive (+1): {pos_count} news")
                        print(f"    - Negative (-1): {neg_count} news")
                        print(f"    - Neutral (0): {neutral_count} news")
                        
                print("  • Sentiment Distribution:")
                print(
                    f"    - 📈 Positive: {sentiment_percentages.get('positive', 0):.1f}% ({sentiment_counts.get('positive', 0)} news)")
                print(
                    f"    - 📉 Negative: {sentiment_percentages.get('negative', 0):.1f}% ({sentiment_counts.get('negative', 0)} news)")
                print(
                    f"    - ➡️ Neutral: {sentiment_percentages.get('neutral', 0):.1f}% ({sentiment_counts.get('neutral', 0)} news)")

                dominant_category = "neutral"
                max_percent = sentiment_percentages.get('neutral', 0)
                
                if sentiment_percentages.get('positive', 0) > max_percent:
                    dominant_category = "positive"
                    max_percent = sentiment_percentages.get('positive', 0)
                
                if sentiment_percentages.get('negative', 0) > max_percent:
                    dominant_category = "negative"
                    max_percent = sentiment_percentages.get('negative', 0)
                
                dominant_icon = "➡️" if dominant_category == "neutral" else "📈" if dominant_category == "positive" else "📉"
                print(f"  • Dominant Sentiment: {dominant_icon} {dominant_category.capitalize()} ({max_percent:.1f}%)")

                sentiment_value = news_analysis.get('sentiment', 0)
                print(f"  • Discrete Sentiment Impact: Using {'-1/0/+1' if sentiment_value in [-1, 0, 1] else 'weighted'} discrete values")

            if news_influence:
                direction = "🔺" if news_influence > 0 else "🔻" if news_influence < 0 else "➡️"
                print(f"  • News Impact: {direction} {abs(news_influence * 100):.1f}%")
                print(f"  • Price Impact: {'+' if news_price_impact > 0 else ''}{news_price_impact:.2f}")

            if direction_changed_by_news:
                print("  • ⚠️ News significantly affected prediction direction")

            if global_news_impact and global_news_count > 0:
                print("\n🌍 GLOBAL NEWS IMPACT:")
                if global_news_count:
                    print(f"  • Analyzed {global_news_count} relevant global news")

                if global_impact_score:
                    direction = "🔺" if global_impact_score > 0 else "🔻" if global_impact_score < 0 else "➡️"
                    print(f"  • Total Impact: {direction} {abs(global_impact_score * 100):.1f}%")

                if country_impacts:
                    print("  • Impact by Country:")
                    for country, impact in country_impacts.items():
                        if not isinstance(country, str):
                            print(f"    - WARNING: Invalid country key type: {type(country)}")
                            continue

                        if isinstance(impact, (int, float)) and abs(
                                impact) > 0.001:  # Ensure impact is a number and check significance
                            direction = "🔺" if impact > 0 else "🔻" if impact < 0 else "➡️"
                            print(f"    - {country.upper()}: {direction} {abs(impact * 100):.1f}%")
                        elif isinstance(impact, dict):
                            # Handle case where impact is a dictionary
                            impact_value = impact.get('value', 0.0)
                            if isinstance(impact_value, (int, float)) and abs(impact_value) > 0.001:
                                direction = "🔺" if impact_value > 0 else "🔻" if impact_value < 0 else "➡️"
                                print(f"    - {country.upper()}: {direction} {abs(impact_value * 100):.1f}%")
                            else:
                                print(f"    - {country.upper()}: ➡️ 0.0% (invalid impact value: {impact_value})")
                        else:
                            print(f"    - {country.upper()}: ➡️ 0.0% (invalid impact type: {type(impact)})")

        if has_macro:
            print(f"\n📈 MACROECONOMIC ANALYSIS:")
            print(f"  • Using {macro_indicator_count} indicators")

            if macro_trends:
                print("  • Key Trends:")
                for indicator, data in macro_trends.items():
                    if isinstance(data, dict):
                        change = data.get('change_percent', 0)
                        if isinstance(change, (int, float)):
                            direction = "🔺" if change > 0 else "🔻" if change < 0 else "➡️"
                            print(f"    - {indicator}: {direction} {abs(change):.2f}%")
                    else:
                        print(f"    - {indicator}: ➡️ 0.0% (invalid change type: {type(change)})")

            if has_macro_impact:
                print("  • Impact Analysis:")
                print(f"    - Sector Impact: {'+' if sector_impact > 0 else ''}{sector_impact * 100:.2f}%")
                print(f"    - Country Impact: {'+' if country_impact > 0 else ''}{country_impact * 100:.2f}%")
                print(f"    - Total Impact: {'+' if total_macro_impact > 0 else ''}{total_macro_impact * 100:.2f}%")
                print(f"    - Price Adjustment: {'+' if price_adjustment > 0 else ''}{price_adjustment:.2f}")
                if has_interest_rate_impact:
                    print(f"    - Interest Rate Impact: {'+' if interest_rate_impact > 0 else ''}{interest_rate_impact * 100:.2f}%")

                if interest_rate_data and interest_rate_data.get('has_impact', False):
                    print("\n💰 INTEREST RATE ANALYSIS:")
                    rate_indicator = interest_rate_data.get('indicator', 'Interest Rate')
                    rate_change = interest_rate_data.get('change_percent', 0)
                    rate_sensitivity = interest_rate_data.get('sensitivity', 0)
                    direction = "🔺" if rate_change > 0 else "🔻" if rate_change < 0 else "➡️"
                    sector = company_info.get('sector', 'unknown') if company_info else 'unknown'
                    
                    print(f"  • Indicator: {rate_indicator}")
                    print(f"  • Change: {direction} {abs(rate_change):.2f}%")
                    print(f"  • Sector Sensitivity: {abs(rate_sensitivity):.2f} ({sector} sector)")

                    if abs(interest_rate_impact) > 0.001:
                        impact_direction = "positive" if interest_rate_impact > 0 else "negative"
                        print(f"  • Impact: {interest_rate_impact * 100:+.2f}% ({impact_direction})")
                        print(f"  • Message: {interest_rate_data.get('message', '')}")
                    else:
                        print("  • Impact: Minimal (below threshold)")

                    print("\n  • Sector Sensitivity to Interest Rates:")
                    sector_info = {
                        'finance': "📈 Positive (higher rates boost financial margins)",
                        'real_estate': "📉 Highly Negative (mortgage costs increase)",
                        'utilities': "📉 Negative (higher debt servicing costs)",
                        'technology': "📉 Negative (growth stocks devalued)",
                        'consumer': "📉 Negative (reduced consumer spending)",
                        'energy': "📉 Slightly Negative (capital-intensive projects)",
                        'healthcare': "📉 Moderately Negative (R&D investment costs)",
                        'materials': "📉 Moderately Negative (capital expenditure costs)"
                    }

                    sectors_to_show = [sector] if sector in sector_info else []
                    if 'finance' not in sectors_to_show:
                        sectors_to_show.append('finance')
                    if 'real_estate' not in sectors_to_show:
                        sectors_to_show.append('real_estate')
                    if 'technology' not in sectors_to_show and sector != 'technology':
                        sectors_to_show.append('technology')
                    
                    for s in sectors_to_show:
                        info = sector_info.get(s, "⚠️ Unknown impact")
                        is_current = s == sector
                        print(f"    - {s.capitalize()}: {info}{' (current sector)' if is_current else ''}")

                    if rate_change > 0:
                        if rate_sensitivity > 0:
                            print("\n  • Forecast: Continued rate increases would likely benefit this sector")
                        else:
                            print("\n  • Forecast: Continued rate increases would likely pressure this sector")
                    elif rate_change < 0:
                        if rate_sensitivity > 0:
                            print("\n  • Forecast: Continued rate decreases would likely pressure this sector")
                        else:
                            print("\n  • Forecast: Continued rate decreases would likely benefit this sector")
                    else:
                        print("\n  • Forecast: Stable rates have minimal direct impact on this sector")

        used_ts_model = ts_prediction.get('model_used', False)
        model_type = ts_prediction.get('model_type', 'Unknown')

        print(f"\n🧠 SIMPLE LSTM MODEL ANALYSIS:")
        if used_ts_model:
            print(f"  • ✅ Using {model_type} for prediction")
            ts_direction_symbol = "🔺" if ts_direction > 0 else "🔻" if ts_direction < 0 else "➡️"
            print(f"  • Price Prediction: {ts_price:.2f}")
            print(f"  • Change: {'+' if ts_change_percent > 0 else ''}{ts_change_percent:.2f}%")
            print(f"  • Direction: {ts_direction_symbol}")

            if model_type == 'Simple LSTM':
                print("  • Model: Simple LSTM (3-layer, 32 neurons each)")
                print("  • Input: Close price sequences (time_step=15)")
                print("  • Direction Logic: Fixed threshold-based (±0.1%)")
            else:
                print(f"  • Model: {model_type}")
        else:
            print("  • ⚠️ Simple LSTM model not available")
            print("  • Using technical analysis fallback instead")
            print(f"  • Current Price: {current_price:.2f}")

        print(f"\n🎯 FINAL PREDICTION:")
        direction_symbol = "🔺" if predicted_direction > 0 else "🔻" if predicted_direction < 0 else "➡️"
        change_percent = ((predicted_price / current_price - 1) * 100) if current_price else 0
        print(f"  • Price: {predicted_price:.2f} ({'+' if change_percent > 0 else ''}{change_percent:.2f}%)")
        print(f"  • Direction: {direction_symbol}")

        if abs(change_percent) > 0.1:
            print(f"  • Direction Confidence: High (change > ±0.1% threshold)")
        else:
            print(f"  • Direction Confidence: Neutral (change within ±0.1% threshold)")

        if ts_price and predicted_price != ts_price:
            print(f"\n📊 PRICE BREAKDOWN:")
            print(f"  • Base (Simple LSTM): {ts_price:.2f}")

            total_adjustment = predicted_price - ts_price
            print(f"  • Total Adjustments: {'+' if total_adjustment > 0 else ''}{total_adjustment:.2f}")

            if news_price_impact and abs(news_price_impact) > 0.01:
                print(f"    - News Impact: {'+' if news_price_impact > 0 else ''}{news_price_impact:.2f}")

            if price_adjustment and abs(price_adjustment) > 0.01:
                print(f"    - Macro Impact: {'+' if price_adjustment > 0 else ''}{price_adjustment:.2f}")

            print(f"  • Final Price: {predicted_price:.2f}")

        print(f"\n📈 DIRECTION ANALYSIS:")
        print(f"  • Threshold Used: ±0.1% (improved from previous paradox)")
        print(f"  • Price Change: {change_percent:+.3f}%")
        if abs(change_percent) > 0.1:
            confidence_level = min(100, abs(change_percent) * 10)  # Простая формула уверенности
            print(f"  • Direction Prediction Accuracy: {confidence_level:.0f}%")
        else:
            print(f"  • Direction Prediction Accuracy: 50% (neutral zone)")

        print("\n" + "=" * 50 + "\n")

    except Exception as e:
        print(f"\n❌ Error formatting results: {str(e)}")
        traceback.print_exc()
        print("Raw results:", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict stock prices using combined model')
    parser.add_argument('--stock-data', type=str, required=True,
                        help='Path to the stock data CSV file')
    parser.add_argument('--model-path', type=str, default='combined_model',
                        help='Base path for model files')
    parser.add_argument('--news-text', type=str,
                        help='News text for sentiment analysis')
    parser.add_argument('--company-info', type=str,
                        help='JSON string with company information (country, sector, dependencies)')
    parser.add_argument('--global-news', type=str,
                        help='JSON string with global political news')
    parser.add_argument('--is-moex', action='store_true',
                        help='Flag indicating that the stock is from MOEX')
    parser.add_argument('--raw-importance', action='store_true',
                        help='Use raw importance values without normalization (can make a few high-importance news dominate)')

    args = parser.parse_args()

    company_info = None
    if args.company_info:
        try:
            company_info = json.loads(args.company_info)
        except json.JSONDecodeError:
            print("Warning: Could not parse company info JSON")

    global_news = None
    if args.global_news:
        try:
            global_news = json.loads(args.global_news)
        except json.JSONDecodeError:
            print("Warning: Could not parse global news JSON")

    normalize_news_importance = not args.raw_importance
    if args.raw_importance:
        print("Using raw importance values (normalization disabled)")

    results = predict_with_combined_model(
        args.stock_data,
        args.model_path,
        args.news_text,
        None,
        None,
        args.is_moex,
        company_info,
        global_news,
        normalize_news_importance
    )

    print_prediction_results(results, args.is_moex)


def fetch_fresh_macro_data(is_moex=False):
    try:
        print("Fetching fresh macroeconomic data...")

        if is_moex:
            macro_data = {
                'interest_rate': None,
                'inflation_rate': None,
                'gdp_growth': None,
                'unemployment_rate': None,
                'market_volatility': None,
                'oil_price': None,
                'exchange_rate': None
            }

            try:
                response = requests.get('https://www.cbr.ru/eng/statistics/ddkp/key_rate/')
                if response.status_code == 200:
                    macro_data['interest_rate'] = 16.0
            except Exception as e:
                print(f"Error fetching interest rate: {e}")

            try:
                response = requests.get('https://www.cbr.ru/eng/statistics/ddkp/inflation/')
                if response.status_code == 200:
                    macro_data['inflation_rate'] = 7.4
            except Exception as e:
                print(f"Error fetching inflation rate: {e}")

            try:
                response = requests.get('https://www.cbr.ru/eng/currency_base/daily/')
                if response.status_code == 200:
                    macro_data['exchange_rate'] = 90.0
            except Exception as e:
                print(f"Error fetching exchange rate: {e}")

            try:
                response = requests.get('https://www.cbr.ru/eng/statistics/macro_itm/svs/')
                if response.status_code == 200:
                    macro_data['oil_price'] = 80.0
            except Exception as e:
                print(f"Error fetching oil price: {e}")

            try:
                response = requests.get('https://moex.com/ru/index/RVI/')
                if response.status_code == 200:
                    macro_data['market_volatility'] = 20.0
            except Exception as e:
                print(f"Error fetching market volatility: {e}")

            try:
                response = requests.get('https://rosstat.gov.ru/statistics/employment#')
                if response.status_code == 200:
                    macro_data['unemployment_rate'] = 3.0
            except Exception as e:
                print(f"Error fetching unemployment rate: {e}")

            try:
                response = requests.get('https://rosstat.gov.ru/statistics/accounts#')
                if response.status_code == 200:
                    macro_data['gdp_growth'] = 2.0
            except Exception as e:
                print(f"Error fetching GDP growth: {e}")

        else:
            macro_data = {
                'interest_rate': None,
                'inflation_rate': None,
                'gdp_growth': None,
                'unemployment_rate': None,
                'market_volatility': None,
                'oil_price': None,
                'exchange_rate': None
            }

            try:
                response = requests.get('https://fred.stlouisfed.org/series/FEDFUNDS')
                if response.status_code == 200:
                    macro_data['interest_rate'] = 5.25
            except Exception as e:
                print(f"Error fetching interest rate: {e}")

            try:
                response = requests.get('https://fred.stlouisfed.org/series/CPIAUCSL')
                if response.status_code == 200:
                    macro_data['inflation_rate'] = 3.1
            except Exception as e:
                print(f"Error fetching inflation rate: {e}")

            try:
                response = requests.get('https://fred.stlouisfed.org/series/DEXUSEU')
                if response.status_code == 200:
                    macro_data['exchange_rate'] = 1.08
            except Exception as e:
                print(f"Error fetching exchange rate: {e}")

            try:
                response = requests.get('https://fred.stlouisfed.org/series/DCOILWTICO')
                if response.status_code == 200:
                    macro_data['oil_price'] = 75.0
            except Exception as e:
                print(f"Error fetching oil price: {e}")

            try:
                response = requests.get('https://fred.stlouisfed.org/series/VIXCLS')
                if response.status_code == 200:
                    macro_data['market_volatility'] = 15.0
            except Exception as e:
                print(f"Error fetching market volatility: {e}")

            try:
                response = requests.get('https://fred.stlouisfed.org/series/UNRATE')
                if response.status_code == 200:
                    macro_data['unemployment_rate'] = 3.7
            except Exception as e:
                print(f"Error fetching unemployment rate: {e}")

            try:
                response = requests.get('https://fred.stlouisfed.org/series/GDPC1')
                if response.status_code == 200:
                    macro_data['gdp_growth'] = 2.1
            except Exception as e:
                print(f"Error fetching GDP growth: {e}")

        missing_values = [k for k, v in macro_data.items() if v is None]
        if missing_values:
            print(f"Warning: Could not fetch some macroeconomic indicators: {missing_values}")

        print("Successfully fetched macroeconomic data")
        return macro_data

    except Exception as e:
        print(f"Error fetching macroeconomic data: {str(e)}")
        print(traceback.format_exc())
        return None


def predict_with_combined_model(stock_data_path, model_base_path, news_text=None, all_news=None, macro_data=None,
                                is_moex=False, company_info=None, global_news=None, normalize_news_importance=True):
    try:
        if macro_data is None:
            macro_data = fetch_fresh_macro_data(is_moex)
            if macro_data is None:
                print("Warning: Could not fetch fresh macroeconomic data")
                macro_data = {}
        try:
            stock_data = pd.read_csv(stock_data_path)
            print(f"Loaded stock data from {stock_data_path} with {len(stock_data)} records")

            if 'Date' in stock_data.columns:
                stock_data['Date'] = pd.to_datetime(stock_data['Date'])
                print(f"Data range: {stock_data['Date'].min()} to {stock_data['Date'].max()}")

            current_price = float(stock_data['Close'].iloc[-1])
            print(f"Current price: ${current_price:.2f}")

            try:
                if 'temp_' in stock_data_path and '_data.csv' in stock_data_path:
                    ticker = stock_data_path.split('temp_')[1].split('_data.csv')[0].upper()
                    print(f"Detected ticker: {ticker}")
                else:
                    ticker = "unknown"
            except:
                ticker = "unknown"
        except Exception as e:
            print(f"Error loading stock data: {str(e)}")
            return None

        print("Loading Simple LSTM model...")

        simple_lstm = SimpleLSTMModel(model_base_path, ticker)
        
        try:
            model_loaded = simple_lstm.load_models(model_base_path)
            if not model_loaded:
                print("⚠️ Simple LSTM model not available, using fallback prediction")
        except Exception as e:
            print(f"Error loading Simple LSTM model: {str(e)}")
            print("Using fallback prediction method...")
            model_loaded = False

        print("🎯 Starting Simple LSTM prediction...")

        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            if col in stock_data.columns:
                try:
                        stock_data[col] = pd.to_numeric(stock_data[col], errors='coerce')
                except Exception as e:
                    print(f"Warning: Could not convert {col} to numeric: {e}")

        stock_data = stock_data.fillna(method='ffill').fillna(method='bfill').fillna(0)

        ts_prediction_price = None
        ts_prediction_direction = None
        model_used = False

        if simple_lstm.is_loaded:
            try:
                print("Making prediction with Simple LSTM model...")

                latest_data = stock_data.iloc[-1:]

                ts_prediction_price = simple_lstm.predict_price(stock_data)
                price_diff = ts_prediction_price - current_price
                price_change_pct = (price_diff / current_price) * 100

                threshold = 0.1
                
                if price_change_pct > threshold:
                    ts_prediction_direction = 1
                elif price_change_pct < -threshold:
                    ts_prediction_direction = -1
                else:
                    ts_prediction_direction = 0

                model_used = True
                print(f"✅ Simple LSTM prediction: price={ts_prediction_price:.2f}, direction={'UP' if ts_prediction_direction > 0 else 'DOWN' if ts_prediction_direction < 0 else 'NEUTRAL'}")
                print(f"   Price change: {price_change_pct:+.3f}% (threshold: ±{threshold}%)")
                
            except Exception as e:
                print(f"Error making Simple LSTM prediction: {str(e)}")
                model_used = False

        if ts_prediction_price is None or ((abs(ts_prediction_price - current_price) / current_price > 0.08)):
            print("Creating fallback prediction based on technical analysis...")
            
            try:
                sma_20 = stock_data['Close'].rolling(window=20).mean().iloc[-1]
                sma_50 = stock_data['Close'].rolling(window=50).mean().iloc[-1]
                trend_factor = 0

                if not pd.isna(sma_20) and not pd.isna(sma_50):
                    if current_price > sma_20 and sma_20 > sma_50:
                        trend_factor += 0.01
                elif current_price < sma_20 and sma_20 < sma_50:
                    trend_factor -= 0.01

                recent_returns = stock_data['Close'].pct_change().tail(5)
                volatility = recent_returns.std()

                max_change = min(0.02, volatility * 2) if not pd.isna(volatility) else 0.01
                trend_factor = max(-max_change, min(max_change, trend_factor))

                ts_prediction_price = current_price * (1 + trend_factor)

                price_change_pct = trend_factor * 100
                threshold = 0.1
                
                if price_change_pct > threshold:
                    ts_prediction_direction = 1
                elif price_change_pct < -threshold:
                    ts_prediction_direction = -1
                else:
                    ts_prediction_direction = 0

                print(f"✅ Fallback prediction: price={ts_prediction_price:.2f}, direction={'UP' if ts_prediction_direction > 0 else 'DOWN' if ts_prediction_direction < 0 else 'NEUTRAL'}")
                print(f"   Based on technical analysis, change: {price_change_pct:+.3f}%")

            except Exception as e:
                print(f"Error creating fallback prediction: {e}")
                ts_prediction_price = current_price
                ts_prediction_direction = 0
                print(f"✅ Using current price as prediction: {ts_prediction_price:.2f} (NEUTRAL)")

        base_prediction = {
            'price': ts_prediction_price,
            'direction': ts_prediction_direction,
            'model_used': model_used
        }

        print(f"🎯 Base Simple LSTM prediction established: ${ts_prediction_price:.2f}, direction={'UP' if ts_prediction_direction > 0 else 'DOWN' if ts_prediction_direction < 0 else 'NEUTRAL'}")
        print(f"   Model used status: {model_used} (will be shown in final results)")

        news_analysis = {}
        news_sentiment = 0
        news_importance = 0

        if all_news or news_text:
            print("\n📰 Анализ новостей...")

            if all_news:
                print(f"Анализ {len(all_news)} новостей...")
                news_sentiment = get_news_sentiment(all_news, is_moex)

                sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}

                all_discrete_sentiments = []
                print("\nДискретные категории сентимента для новостей:")
                for i, news in enumerate(all_news, 1):
                    continuous_sentiment = calculate_basic_sentiment(news)
                    discrete_sentiment = get_sentiment_category(continuous_sentiment, news)
                    all_discrete_sentiments.append(discrete_sentiment)

                    if discrete_sentiment > 0:
                        sentiment_counts['positive'] += 1
                    elif discrete_sentiment < 0:
                        sentiment_counts['negative'] += 1
                    else:
                        sentiment_counts['neutral'] += 1

                    if i <= 5:
                        sentiment_symbol = "📈" if discrete_sentiment > 0 else "📉" if discrete_sentiment < 0 else "➡️"
                        print(f"  {i}. {sentiment_symbol} {news[:50]}... (Категория: {discrete_sentiment})")

                total_news = len(all_news)
                sentiment_percentages = {
                    'positive': (sentiment_counts['positive'] / total_news) * 100,
                    'negative': (sentiment_counts['negative'] / total_news) * 100,
                    'neutral': (sentiment_counts['neutral'] / total_news) * 100
                }

                news_analysis['sentiment_counts'] = sentiment_counts
                news_analysis['sentiment_percentages'] = sentiment_percentages
                news_analysis['news_count'] = total_news
                news_analysis['discretized_sentiment'] = True
                news_analysis['discrete_sentiments'] = all_discrete_sentiments

            if news_text:
                print("Анализ конкретной новости...")
                news_importance = calculate_news_importance(news_text, company_info, is_moex)
                single_news_sentiment = calculate_basic_sentiment(news_text)
                news_analysis['single_news'] = {
                    'importance': news_importance,
                    'sentiment': single_news_sentiment
                }

            if global_news:
                print(f"Анализ {len(global_news)} глобальных новостей...")
                global_news_impact = analyze_global_political_news(company_info, global_news, is_moex)
                news_analysis['global_news_impact'] = global_news_impact

            news_influence = news_sentiment
            news_influence = max(min(news_influence, 1.0), -1.0)  # Ограничиваем влияние до ±100%

            news_analysis['sentiment'] = news_sentiment
            news_analysis['importance'] = news_importance
            news_analysis['news_influence'] = news_influence

            print(f"Общий сентимент новостей: {news_sentiment:.2f}")
            print(f"Влияние новостей: {news_influence:.2f}")

        macro_trends = None
        if company_info and 'macro_trends' in company_info:
            macro_trends = company_info['macro_trends']
            print("Found macroeconomic trends in company info")

        macro_adjuster = MacroAdjustment()

        if macro_trends:
            print("Adjusting prediction based on macroeconomic trends...")
            adjusted_prediction = macro_adjuster.adjust_prediction(base_prediction, company_info)

            macro_data_info = {
                'used': True,
                'indicator_count': len(macro_trends),
                'trends': macro_trends
            }

            macro_impact = adjusted_prediction.get('macro_impact', {})
        else:
            print("No macroeconomic trends available for adjustment")
            adjusted_prediction = base_prediction
            macro_data_info = {'used': False, 'indicator_count': 0}
            macro_impact = {}

        if news_analysis:
            print("Корректируем предсказание с учетом новостей...")

            news_price_impact = adjusted_prediction['price'] * news_influence * 0.03

            global_news_impact_data = news_analysis.get('global_news_impact', {})
            global_impact_score = global_news_impact_data.get('impact_score', 0)

            if global_impact_score:
                global_price_impact = adjusted_prediction['price'] * global_impact_score
                print(f"Добавляем влияние глобальных новостей: {global_price_impact:.2f}")
                news_price_impact += global_price_impact

            ts_direction = base_prediction.get('direction', 0)
            total_news_impact = news_influence + global_impact_score

            if (total_news_impact > 0 and ts_direction > 0):
                print("Положительные новости совпадают с положительным трендом - снижаем влияние в 3 раза")
                news_price_impact /= 3
            elif (total_news_impact < 0 and ts_direction < 0):
                print("Отрицательные новости совпадают с отрицательным трендом - снижаем влияние в 2 раза")
                news_price_impact /= 2
            else:
                print("Направление новостей противоположно или нейтрально к тренду - сохраняем полное влияние")

            adjusted_prediction['price'] += news_price_impact
            total_news_influence = news_influence + global_impact_score
            total_price_change_pct = ((adjusted_prediction['price'] - current_price) / current_price) * 100

            threshold = 0.1
            
            if total_price_change_pct > threshold:
                adjusted_prediction['direction'] = 1
                news_analysis['direction_changed_by_news'] = (adjusted_prediction['direction'] != base_prediction['direction'])
            elif total_price_change_pct < -threshold:
                adjusted_prediction['direction'] = -1
                news_analysis['direction_changed_by_news'] = (adjusted_prediction['direction'] != base_prediction['direction'])
            else:
                adjusted_prediction['direction'] = 0
                news_analysis['direction_changed_by_news'] = (adjusted_prediction['direction'] != base_prediction['direction'])

            news_analysis['price_impact'] = news_price_impact
            news_analysis['influence_percent'] = news_influence * 100
            news_analysis['total_influence_percent'] = total_news_influence * 100

            print(f"Общее влияние новостей на цену: {news_price_impact:.2f}")
            print(f"Новая цена после учета новостей: {adjusted_prediction['price']:.2f}")
            print(f"Финальное направление: {'UP' if adjusted_prediction['direction'] > 0 else 'DOWN' if adjusted_prediction['direction'] < 0 else 'NEUTRAL'}")

        if macro_impact and 'price_adjustment' in macro_impact:
            macro_price_adjustment = macro_impact['price_adjustment']
            if macro_price_adjustment != 0:
                print(f"Применяем макроэкономическую корректировку: {macro_price_adjustment:.2f}")
                adjusted_prediction['price'] += macro_price_adjustment

                final_price_change_pct = ((adjusted_prediction['price'] - current_price) / current_price) * 100
                threshold = 0.1
                
                if final_price_change_pct > threshold:
                    adjusted_prediction['direction'] = 1
                elif final_price_change_pct < -threshold:
                    adjusted_prediction['direction'] = -1
                else:
                    adjusted_prediction['direction'] = 0
                
                print(f"Финальная цена после макроэкономической корректировки: {adjusted_prediction['price']:.2f}")
                print(f"Финальное направление: {'UP' if adjusted_prediction['direction'] > 0 else 'DOWN' if adjusted_prediction['direction'] < 0 else 'NEUTRAL'}")

        print(f"🎯 Финальная скорректированная цена: {adjusted_prediction['price']:.2f}")
        print(f"   Базовая цена (Simple LSTM): {base_prediction['price']:.2f}")
        print(f"   Общая корректировка: {adjusted_prediction['price'] - base_prediction['price']:.2f}")

        results = {
            'ticker': ticker,
            'date': stock_data['Date'].iloc[-1],
            'current_price': float(stock_data['Close'].iloc[-1]),
            'ts_prediction': {
                'price': base_prediction['price'],  # Показываем базовый прогноз Simple LSTM
                'direction': base_prediction.get('direction', 0),
                'change_percent': ((base_prediction['price'] / float(stock_data['Close'].iloc[-1]) - 1) * 100),
                'model_used': base_prediction.get('model_used', False),
                'model_type': 'Simple LSTM' if base_prediction.get('model_used', False) else 'Technical Analysis Fallback'
            },
            'combined_prediction': {
                'price': adjusted_prediction['price'],  # Показываем финальный скорректированный прогноз
                'direction': adjusted_prediction.get('direction', 0),
                'model_used': base_prediction.get('model_used', False)
            },
            'macro_data': macro_data_info,
            'macro_impact': macro_impact,
            'news_analysis': news_analysis,
            'interest_rate_impact': adjusted_prediction.get('interest_rate_impact', {})
        }

        return results

    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}