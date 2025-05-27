import os

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
import argparse
import sys
import yfinance as yf
from datetime import datetime, timedelta
import requests
import json
from typing import Dict, List, Optional, Tuple
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from combined_model import CombinedPredictionModel
from update_data import update_stock_data, get_news
from macro_data import MacroDataLoader

MOEX_TICKERS = {
    'SBER': 'Сбербанк',
    'GAZP': 'Газпром',
    'LKOH': 'Лукойл',
    'YNDX': 'Яндекс',
    'TCSG': 'Тинькофф Банк',
    'ROSN': 'Роснефть',
    'NVTK': 'Новатэк',
    'TATN': 'Татнефть',
    'ALRS': 'Алроса',
    'POLY': 'Полюс',
    'RUAL': 'РУСАЛ',
    'MAGN': 'Магнитогорский металлургический комбинат',
    'NLMK': 'Новолипецкий металлургический комбинат',
    'CHMF': 'Северсталь',
    'PHOR': 'ФосАгро',
    'PLZL': 'Полюс Золото',
    'RASP': 'Распадская',
    'AFLT': 'Аэрофлот',
    'MOEX': 'Московская биржа',
    'VTBR': 'ВТБ',
    'FIVE': 'X5 Group',
    'OZON': 'Ozon',
    'VKUS': 'ВкусВилл',
    'QIWI': 'QIWI',
    'MGNT': 'Магнит',
    'FIXP': 'Fix Price',
    'TCS': 'Тинькофф',
    'VKUS': 'ВкусВилл',
    'DSKY': 'Детский мир',
    'LENT': 'Лента',
    'MVID': 'М.Видео',
    'POSI': 'Позитив Технолоджис',
    'RENI': 'Ренессанс Страхование',
    'SELG': 'Селигдар',
    'SFTL': 'Совкомфлот',
    'UPRO': 'Юнипро',
    'VRSB': 'ТМК',
    'WUSH': 'ВкусВилл',
    'YAKG': 'Якутскэнерго'
}

SECTORS = {
    'SBER': 'finance_ru',
    'GAZP': 'energy_ru',
    'LKOH': 'energy_ru',
    'YNDX': 'technology_ru',
    'TCSG': 'finance_ru',
    'ROSN': 'energy_ru',
    'NVTK': 'energy_ru',
    'TATN': 'energy_ru',
    'ALRS': 'materials_ru',
    'POLY': 'materials_ru',
    'RUAL': 'materials_ru',
    'MAGN': 'materials_ru',
    'NLMK': 'materials_ru',
    'CHMF': 'materials_ru',
    'PHOR': 'materials_ru',
    'PLZL': 'materials_ru',
    'RASP': 'materials_ru',
    'AFLT': 'transportation_ru',
    'MOEX': 'finance_ru',
    'VTBR': 'finance_ru',
    'FIVE': 'consumer_ru',
    'OZON': 'technology_ru',
    'VKUS': 'consumer_ru',
    'QIWI': 'finance_ru',
    'MGNT': 'consumer_ru',
    'FIXP': 'consumer_ru',
    'DSKY': 'consumer_ru',
    'LENT': 'consumer_ru',
    'MVID': 'consumer_ru',
    'POSI': 'technology_ru',

    'AAPL': 'technology',
    'MSFT': 'technology',
    'AMZN': 'technology',
    'GOOGL': 'technology',
    'GOOG': 'technology',
    'META': 'technology',
    'TSLA': 'automotive',
    'NVDA': 'technology',
    'JPM': 'finance',
    'V': 'finance',
    'PG': 'consumer',
    'UNH': 'healthcare',
    'HD': 'consumer',
    'BAC': 'finance',
    'XOM': 'energy',
    'AVGO': 'technology',
    'MA': 'finance',
    'CVX': 'energy',
    'COST': 'consumer',
    'ABBV': 'healthcare',
    'PEP': 'consumer',
    'KO': 'consumer',
    'WMT': 'consumer',
    'MRK': 'healthcare',
    'LLY': 'healthcare',
    'CSCO': 'technology',
    'TMO': 'healthcare',
    'ADBE': 'technology',
    'MCD': 'consumer',
    'ABT': 'healthcare',
    'CRM': 'technology',
    'DHR': 'healthcare',
    'ACN': 'technology',
    'NKE': 'consumer',
    'DIS': 'communication',
    'NEE': 'utilities',
    'AMD': 'technology',
    'PM': 'consumer',
    'TXN': 'technology',
    'INTC': 'technology',
    'IBM': 'technology',
    'QCOM': 'technology',
    'NFLX': 'communication',
    'PYPL': 'finance',
    'SBUX': 'consumer',
    'INTU': 'technology',
    'AMAT': 'technology',
    'BKNG': 'consumer',
    'CAT': 'industrial',
    'BA': 'industrial',
    'GE': 'industrial',
    'F': 'automotive',
    'GM': 'automotive',
    'UBER': 'technology',
    'LYFT': 'technology',
    'COIN': 'finance',
    'PLTR': 'technology',
    'SNOW': 'technology',
    'ZM': 'technology'
}

SECTOR_DEPENDENCIES = {
    'energy': [
        {'country': 'usa', 'type': 'market', 'strength': 0.7},
        {'country': 'china', 'type': 'market', 'strength': 0.8},
        {'country': 'eu', 'type': 'market', 'strength': 0.6}
    ],
    'materials': [
        {'country': 'china', 'type': 'market', 'strength': 0.9},
        {'country': 'usa', 'type': 'market', 'strength': 0.6},
        {'country': 'eu', 'type': 'market', 'strength': 0.5}
    ],
    'finance': [
        {'country': 'usa', 'type': 'market', 'strength': 0.8},
        {'country': 'eu', 'type': 'market', 'strength': 0.6}
    ],
    'technology': [
        {'country': 'usa', 'type': 'supplier', 'strength': 0.9},
        {'country': 'china', 'type': 'supplier', 'strength': 0.7},
        {'country': 'taiwan', 'type': 'supplier', 'strength': 0.8}
    ],
    'consumer': [
        {'country': 'china', 'type': 'supplier', 'strength': 0.7},
        {'country': 'usa', 'type': 'market', 'strength': 0.6}
    ],
    'healthcare': [
        {'country': 'usa', 'type': 'supplier', 'strength': 0.8},
        {'country': 'eu', 'type': 'market', 'strength': 0.6}
    ],
    'transportation': [
        {'country': 'usa', 'type': 'supplier', 'strength': 0.6},
        {'country': 'eu', 'type': 'market', 'strength': 0.5}
    ],
    'automotive': [
        {'country': 'china', 'type': 'market', 'strength': 0.8},
        {'country': 'usa', 'type': 'market', 'strength': 0.7},
        {'country': 'eu', 'type': 'market', 'strength': 0.6}
    ],
    'industrial': [
        {'country': 'china', 'type': 'supplier', 'strength': 0.7},
        {'country': 'usa', 'type': 'market', 'strength': 0.8},
        {'country': 'eu', 'type': 'market', 'strength': 0.6}
    ],
    'utilities': [
        {'country': 'usa', 'type': 'market', 'strength': 0.5},
        {'country': 'eu', 'type': 'market', 'strength': 0.4}
    ],
    'communication': [
        {'country': 'usa', 'type': 'market', 'strength': 0.7},
        {'country': 'eu', 'type': 'market', 'strength': 0.6}
    ]
}

NEWS_API_KEY = os.getenv('NEWS_API_KEY', '64134ea289d04936aeb08f00861d9e81')
GLOBAL_NEWS_API_KEY = os.getenv('GLOBAL_NEWS_API_KEY', '64134ea289d04936aeb08f00861d9e81')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

russian_stopwords = set(stopwords.words('russian'))
russian_stemmer = SnowballStemmer('russian')

MOEX_TICKERS_CACHE = {}


def get_company_info_auto(ticker: str, is_moex: bool = False) -> Dict:
    ticker = ticker.upper()

    if is_moex:
        country = 'russia'
    else:
        if ticker.endswith('.L'):
            country = 'uk'
        elif ticker.endswith('.HK'):
            country = 'china'
        elif ticker.endswith('.T'):
            country = 'japan'
        elif ticker.endswith('.F'):
            country = 'germany'
        elif ticker.endswith('.PA'):
            country = 'france'
        else:
            country = 'usa'

    sector = 'unknown'
    base_ticker = ticker.split('.')[0]

    if base_ticker in SECTORS:
        sector = SECTORS[base_ticker]
    else:
        try:
            ticker_info = yf.Ticker(ticker)
            ticker_info_dict = ticker_info.info

            if 'sector' in ticker_info_dict and ticker_info_dict['sector']:
                yf_sector = ticker_info_dict['sector'].lower()

                sector_mapping = {
                    'technology': 'technology',
                    'tech': 'technology',
                    'information technology': 'technology',
                    'financial': 'finance',
                    'financial services': 'finance',
                    'energy': 'energy',
                    'healthcare': 'healthcare',
                    'consumer defensive': 'consumer',
                    'consumer cyclical': 'consumer',
                    'consumer goods': 'consumer',
                    'consumer services': 'consumer',
                    'basic materials': 'materials',
                    'materials': 'materials',
                    'industrial': 'industrial',
                    'industrials': 'industrial',
                    'utilities': 'utilities',
                    'communication services': 'communication',
                    'telecommunications': 'communication',
                    'real estate': 'real_estate',
                    'automotive': 'automotive',
                    'transportation': 'transportation'
                }

                for key, value in sector_mapping.items():
                    if key in yf_sector:
                        sector = value
                        break

                print(f"Found sector from Yahoo Finance: {sector}")

            if country == 'usa' and 'country' in ticker_info_dict and ticker_info_dict['country']:
                yf_country = ticker_info_dict['country'].lower()

                country_mapping = {
                    'united states': 'usa',
                    'usa': 'usa',
                    'russia': 'russia',
                    'russian federation': 'russia',
                    'china': 'china',
                    'united kingdom': 'uk',
                    'uk': 'uk',
                    'japan': 'japan',
                    'germany': 'germany',
                    'france': 'france',
                    'canada': 'canada',
                    'australia': 'australia',
                    'india': 'india',
                    'brazil': 'brazil',
                    'south korea': 'south_korea',
                    'korea': 'south_korea',
                    'taiwan': 'taiwan',
                    'saudi arabia': 'saudi_arabia'
                }

                for key, value in country_mapping.items():
                    if key in yf_country:
                        country = value
                        break

                print(f"Found country from Yahoo Finance: {country}")

        except Exception as e:
            print(f"Error getting info from Yahoo Finance: {str(e)}")

    dependencies = []
    if sector in SECTOR_DEPENDENCIES:
        dependencies = SECTOR_DEPENDENCIES[sector].copy()

        dependencies = [dep for dep in dependencies if dep['country'] != country]

        print(f"Added {len(dependencies)} dependencies based on sector {sector}")

    company_info = {
        'ticker': ticker,
        'country': country,
        'sector': sector,
        'dependencies': dependencies
    }

    print(f"Auto-detected company info: {json.dumps(company_info, indent=2)}")
    return company_info


def get_global_political_news(days: int = 3, max_news: int = 20, company_info: Dict = None) -> List[Dict]:
    print(f"Fetching global political news for the last {days} days...")

    relevant_countries = []
    company_country = None

    if company_info:
        company_country = company_info.get('country', '').lower()
        if company_country:
            relevant_countries.append(company_country)
            print(f"Including news for company's country: {company_country}")

        dependencies = company_info.get('dependencies', [])
        for dependency in dependencies:
            dep_country = dependency.get('country', '').lower()
            if dep_country and dep_country not in relevant_countries:
                relevant_countries.append(dep_country)
                dep_type = dependency.get('type', '')
                dep_strength = dependency.get('strength', 0.5)
                print(
                    f"Including news for dependency country: {dep_country} (type: {dep_type}, strength: {dep_strength:.1f})")

    if not relevant_countries:
        relevant_countries = ['usa', 'china', 'russia', 'eu', 'uk', 'japan', 'india']
        print(f"No company information provided, using major economies: {', '.join(relevant_countries)}")

    country_keywords = {
        'usa': ["USA", "US", "United States", "America", "American", "США", "Америка", "американский"],
        'china': ["China", "Chinese", "Beijing", "Китай", "Китайский", "Пекин"],
        'russia': ["Russia", "Russian", "Moscow", "Россия", "Российский", "Москва"],
        'eu': ["EU", "European Union", "Europe", "European", "ЕС", "Евросоюз", "Европа", "Европейский"],
        'uk': ["UK", "United Kingdom", "Britain", "British", "England", "Великобритания", "Британия", "Британский"],
        'japan': ["Japan", "Japanese", "Tokyo", "Япония", "Японский", "Токио"],
        'india': ["India", "Indian", "Индия", "Индийский"],
        'brazil': ["Brazil", "Brazilian", "Бразилия", "Бразильский"],
        'saudi_arabia': ["Saudi", "Saudi Arabia", "Саудовская Аравия", "Саудовский"],
        'germany': ["Germany", "German", "Berlin", "Германия", "Немецкий", "Берлин"],
        'france': ["France", "French", "Paris", "Франция", "Французский", "Париж"]
    }

    political_keywords = [
        "sanctions", "tariffs", "trade war", "diplomatic", "government", "president",
        "election", "federal reserve", "central bank", "interest rate", "policy",
        "economic policy", "regulation", "sanctions", "embargo", "war", "conflict",
        "санкции", "пошлины", "торговая война", "дипломатический", "правительство", "президент",
        "выборы", "федеральный резерв", "центральный банк", "процентная ставка", "политика",
        "экономическая политика", "регулирование", "санкции", "эмбарго", "война", "конфликт"
    ]

    if company_country:
        sector = company_info.get('sector', '').lower()
        if sector:
            sector_keywords = {
                'energy': ["oil", "gas", "energy", "нефть", "газ", "энергетика", "opec", "опек"],
                'technology': ["chip", "semiconductor", "technology", "чип", "полупроводник", "технологии"],
                'finance': ["bank", "finance", "banking", "банк", "финансы", "банковский"],
                'materials': ["metals", "mining", "materials", "металлы", "добыча", "материалы"],
                'consumer': ["retail", "consumer", "розница", "потребительский"],
                'healthcare': ["health", "medicine", "pharmaceutical", "здоровье", "медицина", "фармацевтика"]
            }

            if sector in sector_keywords:
                for keyword in sector_keywords[sector]:
                    political_keywords.append(keyword)
                print(f"Added sector-specific keywords for {sector}")

    global_news = []

    def add_news_item(text, date_str):
        if not any(item['text'] == text for item in global_news):
            try:
                if isinstance(date_str, str):
                    date_formats = ['%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%d', '%d.%m.%Y', '%m/%d/%Y']
                    parsed_date = None

                    for fmt in date_formats:
                        try:
                            parsed_date = datetime.strptime(date_str, fmt)
                            break
                        except ValueError:
                            continue

                    if parsed_date:
                        date_str = parsed_date.strftime('%Y-%m-%d')
                    else:
                        date_str = datetime.now().strftime('%Y-%m-%d')
                elif isinstance(date_str, datetime):
                    date_str = date_str.strftime('%Y-%m-%d')
            except Exception:
                date_str = datetime.now().strftime('%Y-%m-%d')

            global_news.append({
                'text': text,
                'date': date_str
            })

    try:
        for country_code in relevant_countries:
            country_terms = country_keywords.get(country_code, [country_code.capitalize()])

            for country_term in country_terms[:2]:
                for keyword in political_keywords[:5]:
                    query = f"{keyword} {country_term}"

                    url = 'https://newsapi.org/v2/everything'
                    params = {
                        'q': query,
                        'language': 'en',
                        'sortBy': 'publishedAt',
                        'from': (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
                        'pageSize': 10,
                        'apiKey': GLOBAL_NEWS_API_KEY
                    }

                    try:
                        response = requests.get(url, params=params)
                        response.raise_for_status()

                        news_data = response.json()
                        if news_data['status'] == 'ok' and 'articles' in news_data:
                            for article in news_data['articles']:
                                if len(global_news) >= max_news:
                                    break

                                title = article.get('title', '')
                                if title:
                                    add_news_item(title, article.get('publishedAt', datetime.now()))
                                    print(f"Added news: {title[:50]}...")
                    except Exception as e:
                        print(f"Error fetching news for query '{query}': {str(e)}")
                    if len(global_news) >= max_news:
                        break

                if len(global_news) >= max_news:
                    break

            if len(global_news) >= max_news:
                break
    except Exception as e:
        print(f"Error fetching news from NewsAPI: {str(e)}")

    print(f"Collected {len(global_news)} global political news items relevant to {', '.join(relevant_countries)}")
    return global_news


def get_all_moex_tickers() -> Dict[str, str]:
    global MOEX_TICKERS_CACHE

    if MOEX_TICKERS_CACHE:
        return MOEX_TICKERS_CACHE

    moex_tickers = MOEX_TICKERS.copy()

    try:
        url = "https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities.json"
        response = requests.get(url)
        response.raise_for_status()

        data = response.json()

        if 'securities' in data and 'data' in data['securities']:
            columns = data['securities']['columns']
            secid_idx = columns.index('SECID')
            name_idx = columns.index('SHORTNAME')

            for row in data['securities']['data']:
                ticker = row[secid_idx]
                company_name = row[name_idx]
                moex_tickers[ticker] = company_name

        print(f"Successfully loaded {len(moex_tickers)} tickers from Moscow Exchange")
    except Exception as e:
        print(f"Error loading Moscow Exchange tickers: {str(e)}")
        print("Using predefined ticker list instead")

    MOEX_TICKERS_CACHE = moex_tickers
    return moex_tickers


def is_russian_ticker(ticker: str) -> bool:
    ticker = ticker.upper()

    if ticker in MOEX_TICKERS:
        return True

    all_tickers = get_all_moex_tickers()
    return ticker in all_tickers


def get_company_name(ticker: str, is_moex: bool = False) -> str:
    ticker = ticker.upper()

    if is_moex:
        all_tickers = get_all_moex_tickers()
        return all_tickers.get(ticker, ticker)

    return ticker


def get_russian_news(company_name: str, days: int = 3, max_news: int = 100) -> List[str]:
    try:
        url = 'https://newsapi.org/v2/everything'
        params = {
            'q': company_name,
            'language': 'ru',
            'sortBy': 'publishedAt',
            'pageSize': max_news,
            'apiKey': NEWS_API_KEY
        }

        print(f"Fetching news for {company_name} using NewsAPI...")
        response = requests.get(url, params=params)
        response.raise_for_status()

        news_data = response.json()
        if news_data['status'] == 'ok':
            articles = [article['title'] for article in news_data['articles']]
            print(f"Successfully fetched {len(articles)} news articles from NewsAPI")
            return articles

        print("No articles found in NewsAPI response")
        return []

    except Exception as e:
        print(f"Error fetching news from NewsAPI: {str(e)}")
        print("Falling back to alternative news sources...")
        return get_news_from_alternative_sources(company_name, days, max_news)


def get_news_from_alternative_sources(company_name: str, days: int = 3, max_news: int = 100) -> List[str]:
    try:
        url = 'https://www.rbc.ru/ajax/get_news_feed'
        params = {
            'query': company_name,
            'limit': max_news
        }

        response = requests.get(url, params=params)
        response.raise_for_status()

        news_data = response.json()
        if 'items' in news_data:
            return [item['title'] for item in news_data['items']]

        return []

    except Exception as e:
        print(f"Error fetching news from RBC: {str(e)}")
        return []


def get_news_headlines(symbol: str, days: int = 3, max_news: int = 100, is_moex: bool = False) -> List[str]:
    print(f"Fetching news headlines for {symbol}...")

    if is_moex:
        company_name = get_company_name(symbol, True)
        print(f"Company name: {company_name}")

        news_titles = get_russian_news(company_name, days, max_news)
    else:
        news_titles = get_news(symbol, days, max_news)

    if not news_titles:
        print(f"No news found for {symbol}")
        return []

    print(f"Found {len(news_titles)} news items about {symbol}")
    return news_titles


def get_moex_data(ticker, days=1095):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 100)

        url = f"https://iss.moex.com/iss/engines/stock/markets/shares/securities/{ticker}/trades.json"
        params = {
            'from': start_date.strftime('%Y-%m-%d'),
            'till': end_date.strftime('%Y-%m-%d'),
            'limit': 100000
        }

        print(
            f"Fetching detailed data for {ticker} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

        response = requests.get(url, params=params)
        response.raise_for_status()

        data = response.json()

        if 'trades' not in data or 'data' not in data['trades']:
            print(f"No trade data received for {ticker}")
            return None

        columns = data['trades']['columns']
        trades_data = data['trades']['data']
        trades_df = pd.DataFrame(trades_data, columns=columns)
        trades_df['TRADEDATE'] = pd.to_datetime(trades_df['TRADEDATE'])

        daily_data = trades_df.groupby('TRADEDATE').agg({
            'PRICE': ['first', 'max', 'min', 'last'],
            'VALUE': 'sum'
        }).reset_index()

        daily_data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

        daily_data = daily_data.sort_values('Date')

        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        missing_dates = set(date_range) - set(daily_data['Date'])

        if missing_dates:
            print(f"Fetching additional data for {len(missing_dates)} missing dates...")

            candles_url = f"https://iss.moex.com/iss/engines/stock/markets/shares/securities/{ticker}/candles.json"
            candles_params = {
                'from': min(missing_dates).strftime('%Y-%m-%d'),
                'till': max(missing_dates).strftime('%Y-%m-%d'),
                'interval': 24
            }

            candles_response = requests.get(candles_url, params=candles_params)
            candles_response.raise_for_status()
            candles_data = candles_response.json()

            if 'candles' in candles_data and 'data' in candles_data['candles']:
                candles_columns = candles_data['candles']['columns']

                candles_df = pd.DataFrame(candles_data['candles']['data'], columns=candles_columns)

                candles_df = candles_df.rename(columns={
                    'begin': 'Date',
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                })

                candles_df['Date'] = pd.to_datetime(candles_df['Date'])

                daily_data = pd.concat([daily_data, candles_df], ignore_index=True)
                daily_data = daily_data.sort_values('Date')

        daily_data = daily_data.tail(days)

        print(f"Successfully fetched {len(daily_data)} days of data for {ticker}")
        print(
            f"Latest data point: {daily_data['Date'].max().strftime('%Y-%m-%d')}, Close: {daily_data['Close'].iloc[-1]:.2f}")

        return daily_data

    except Exception as e:
        print(f"Error fetching MOEX data for {ticker}: {str(e)}")
        return None


def predict_with_models(ticker=None, stock_data_path=None, model_base_path="models/simple_lstm", news_text=None,
                        all_news=None, macro_data_path=None, use_macro=False, is_moex=False, company_info=None,
                        global_news=None):

    from predict import predict_with_combined_model, print_prediction_results
    from update_data import update_stock_data
    from macro_data import MacroDataLoader
    import yfinance as yf
    from datetime import datetime, timedelta
    import pandas as pd
    import os
    import tempfile

    print(f"Debug: news_text={news_text is not None}, all_news={len(all_news) if all_news else 'None'}")

    if not company_info and ticker:
        print("Auto-detecting company information...")
        company_info = get_company_info_auto(ticker, is_moex)

    if not global_news:
        print("Fetching global political news...")
        global_news = get_global_political_news(days=3, max_news=20, company_info=company_info)

    if company_info:
        print(
            f"Company info: {company_info.get('ticker', 'unknown')} from {company_info.get('country', 'unknown')}, sector: {company_info.get('sector', 'unknown')}")
        dependencies = company_info.get('dependencies', [])
        if dependencies:
            print("Company dependencies:")
            for dep in dependencies:
                print(
                    f"  - {dep.get('country', 'unknown')} ({dep.get('type', 'unknown')}, strength: {dep.get('strength', 0.5):.1f})")

    if stock_data_path is None and ticker is None:
        raise ValueError("Either stock_data_path or ticker must be provided")

    if ticker is not None and (stock_data_path is None or not os.path.exists(stock_data_path)):
        print(f"Extracting stock data for {ticker}...")

        if is_moex:
            stock_data = get_moex_data(ticker)
        else:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=1095 + 100)

            print(f"Downloading data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
            stock_data = yf.download(ticker, start=start_date, end=end_date)
            stock_data = stock_data.reset_index()

        if stock_data.empty:
            raise ValueError(f"No data found for ticker {ticker}. Please check the ticker symbol.")

        temp_file = f"temp_{ticker}_data.csv"
        stock_data.to_csv(temp_file, index=False)
        print(f"Temporary stock data saved to {temp_file}")

        stock_data_path = temp_file

    if use_macro and company_info:
        print("Анализ ключевых макроэкономических показателей...")
        macro_loader = MacroDataLoader()
        country = company_info.get('country')

        macro_trends = macro_loader.get_key_macro_indicators(days=365, country=country)

        if macro_trends:
            company_info['macro_trends'] = macro_trends
            print("Макроэкономические тренды добавлены в информацию о компании")

    results = predict_with_combined_model(
        stock_data_path,
        model_base_path,
        news_text,
        all_news,
        None,
        is_moex,
        company_info,
        global_news
    )
    print_prediction_results(results, is_moex)

    if ticker is not None and stock_data_path and stock_data_path.startswith("temp_"):
        try:
            os.remove(stock_data_path)
            print(f"Removed temporary file {stock_data_path}")
        except:
            pass

    return results


def update_macro_data(days=360, output_file="macro_data.csv", api_key=None):
    print(f"Updating macroeconomic data for the last {days} days...")

    macro_loader = MacroDataLoader(api_key=api_key)
    macro_data = macro_loader.get_all_macro_features(days=days)

    if macro_data is not None and not macro_data.empty:
        if isinstance(macro_data.index, pd.DatetimeIndex):
            macro_data = macro_data.reset_index().rename(columns={'index': 'Date'})

        macro_data.to_csv(output_file, index=False)
        print(f"Macroeconomic data with {len(macro_data)} records and {len(macro_data.columns)} "
              f"features saved to {output_file}")
        return macro_data
    else:
        print("Failed to retrieve macroeconomic data")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stock Price and Trend Prediction System')
    parser.add_argument('--model-path', type=str, default='models',
                        help='Base path for model files')
    parser.add_argument('--days', type=int, default=3,
                        help='Number of days for news lookup')
    parser.add_argument('--use-macro', action='store_true', default=True,
                        help='Include macroeconomic features in predictions')
    parser.add_argument('--is-moex', action='store_true',
                        help='Force use of MOEX (Moscow Exchange) data for Russian stocks')
    args = parser.parse_args()

    # Запрашиваем тикер у пользователя
    print("\n=== Stock Price and Trend Prediction System ===")
    print("This system combines technical analysis, news sentiment analysis")
    print("and macroeconomic indicators to predict stock price movements.\n")

    symbol = input("Enter stock ticker symbol (e.g., AAPL, MSFT, SBER): ")
    if not symbol:
        symbol = "AAPL"  # значение по умолчанию, если пользователь ничего не ввел
        print(f"No ticker entered, using default: {symbol}")

    print(f"\n🔍 Analyzing stock: {symbol}\n")

    # Автоматически определяем, является ли акция российской
    is_moex = args.is_moex
    if not is_moex:
        is_moex = is_russian_ticker(symbol)
    if is_moex:
        print(f"🇷🇺 Detected Russian stock: {get_company_name(symbol, True)}")
    else:
        print(f"🌐 Detected non-Russian stock: {symbol}")

    # Автоматически определяем информацию о компании
    print("\n📊 Getting company information...")
    company_info = get_company_info_auto(symbol, is_moex)

    # Получаем новости для компании
    print("\n📰 Getting recent news...")
    news_titles = get_news_headlines(symbol, days=args.days, max_news=100, is_moex=is_moex)

    print("\n🌍 Getting global political news...")
    global_news = get_global_political_news(days=args.days, max_news=20, company_info=company_info)

    # Получаем данные акций
    print("\n📈 Fetching stock data...")
    if is_moex:
        stock_data = get_moex_data(symbol)
    else:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1095 + 100)  # 3 года + 100 дней для индикаторов
        print(f"Downloading data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
        stock_data = yf.download(symbol, start=start_date, end=end_date)
        stock_data = stock_data.reset_index()

    if stock_data is not None and not stock_data.empty:
        temp_file = f"temp_{symbol}_data.csv"
        stock_data.to_csv(temp_file, index=False)
        print(f"Stock data saved to temporary file: {temp_file}")

        macro_path = "macro_data.csv"
        if not os.path.exists(macro_path) and args.use_macro:
            print("\n📉 Updating macroeconomic data...")
            update_macro_data(days=360, output_file=macro_path)

        print("\n🔮 Making prediction...")
        predict_with_models(
            symbol,
            temp_file,
            args.model_path,
            None,  # news_text
            news_titles,
            macro_path,
            args.use_macro,
            is_moex,
            company_info,
            global_news
        )

        try:
            os.remove(temp_file)
            print(f"\nRemoved temporary file: {temp_file}")
        except:
            pass
    else:
        print(f"❌ Error: Failed to retrieve stock data for {symbol}")
        sys.exit(1)