import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import argparse
import traceback
import requests
import json
import os
import re
import urllib.parse
import time


def update_stock_data(symbol, output_file, days=360):

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days + 100)

    print(
        f"Fetching stock data for {symbol} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    stock_data = stock_data.reset_index()
    calculate_features(stock_data)
    stock_data = stock_data.tail(days)
    stock_data.to_csv(output_file, index=False)
    print(f"Updated stock data saved to {output_file}")

    return stock_data


def calculate_features(df):
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values('Date', inplace=True)

    df['Returns'] = df['Close'].pct_change()
    df['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['SMA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df.fillna(0, inplace=True)

    return df

def get_news_newsapi(company_name, days=7, max_news=100):
    try:
        print(f"Trying to fetch news using NewsAPI for {company_name}...")
        api_key = os.environ.get('NEWSAPI_KEY', '64134ea289d04936aeb08f00861d9e81')
        if not api_key:
            print("No NewsAPI key found. Set NEWSAPI_KEY environment variable to use this feature.")
            return []

        print(f"Using API key: {api_key[:5]}...{api_key[-5:]}")
        from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        url = "https://newsapi.org/v2/everything"
        params = {
            'q': company_name,
            'from': from_date,
            'sortBy': 'relevancy',
            'language': 'en',
            'pageSize': max_news,
            'apiKey': api_key
        }

        print(f"Sending request to NewsAPI with params: {params}")
        response = requests.get(url, params=params)

        if response.status_code != 200:
            print(f"Error from NewsAPI: {response.status_code} - {response.text}")
            return []

        data = response.json()

        if 'articles' not in data or not data['articles']:
            print(f"No news found for {company_name}")
            return []

        articles = []
        for article in data['articles']:
            articles.append({
                'title': article.get('title', 'No Title'),
                'url': article.get('url', ''),
                'source': article.get('source', {}).get('name', 'Unknown Source'),
                'date': article.get('publishedAt', 'Unknown Date'),
                'content': f"{article.get('description', '')} {article.get('content', '')}"
            })

        print(f"Found {len(articles)} news articles for {company_name} using NewsAPI")
        return articles

    except Exception as e:
        print(f"Error using NewsAPI: {str(e)}")
        print("Detailed error information:")
        traceback.print_exc()
        return []


def get_news_alphavantage(symbol, company_name=None, max_news=50):
    try:
        print(f"Trying to fetch news using Alpha Vantage API for {symbol}...")
        api_key = os.environ.get('ALPHAVANTAGE_KEY', '0QVM37VUGYK5U6W8')
        if not api_key:
            print("No Alpha Vantage API key found. Set ALPHAVANTAGE_KEY environment variable.")
            return []

        print(f"Using Alpha Vantage API key: {api_key[:5]}...{api_key[-5:]}")
        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': symbol,
            'apikey': api_key,
            'limit': max_news
        }

        print(f"Sending request to Alpha Vantage with params: {params}")
        response = requests.get(url, params=params)

        if response.status_code != 200:
            print(f"Error from Alpha Vantage: {response.status_code} - {response.text}")
            return []

        data = response.json()

        if 'feed' not in data or not data['feed']:
            print(f"No news found for {symbol}")
            return []

        articles = []
        for article in data['feed']:
            articles.append({
                'title': article.get('title', 'No Title'),
                'url': article.get('url', ''),
                'source': article.get('source', 'Unknown Source'),
                'date': article.get('time_published', 'Unknown Date'),
                'content': article.get('summary', '')
            })

        print(f"Found {len(articles)} news articles for {symbol} using Alpha Vantage")
        return articles

    except Exception as e:
        print(f"Error using Alpha Vantage API: {str(e)}")
        print("Detailed error information:")
        traceback.print_exc()
        return []


def get_news(symbol, days=7, max_news=150):
    try:
        company_map = {
            'AAPL': 'Apple',
            'MSFT': 'Microsoft',
            'GOOGL': 'Google',
            'AMZN': 'Amazon',
            'TSLA': 'Tesla',
            'META': 'Facebook',
            'NFLX': 'Netflix',
            'IBM': 'IBM',
            'INTC': 'Intel',
            'AMD': 'AMD',
            'NVDA': 'NVIDIA',
            'CSCO': 'Cisco',
            'ORCL': 'Oracle',
            'CRM': 'Salesforce',
            'ADBE': 'Adobe',
            'JPM': 'JPMorgan Chase',
            'BAC': 'Bank of America',
            'WFC': 'Wells Fargo',
            'C': 'Citigroup',
            'GS': 'Goldman Sachs',
            'V': 'Visa',
            'MA': 'Mastercard',
            'PG': 'Procter & Gamble',
            'JNJ': 'Johnson & Johnson',
            'PFE': 'Pfizer',
            'MRNA': 'Moderna',
            'BNTX': 'BioNTech',
            'DIS': 'Disney',
            'NFLX': 'Netflix',
            'CMCSA': 'Comcast',
            'T': 'AT&T',
            'VZ': 'Verizon',
            'TMUS': 'T-Mobile',
            'KO': 'Coca-Cola',
            'PEP': 'PepsiCo',
            'MCD': 'McDonald\'s',
            'SBUX': 'Starbucks',
            'NKE': 'Nike',
            'WMT': 'Walmart',
            'TGT': 'Target',
            'COST': 'Costco',
            'HD': 'Home Depot',
            'LOW': 'Lowe\'s',
            'F': 'Ford',
            'GM': 'General Motors',
            'BA': 'Boeing',
            'LMT': 'Lockheed Martin',
            'GE': 'General Electric',
            'CAT': 'Caterpillar',
            'MMM': '3M',
            'CVX': 'Chevron',
            'XOM': 'Exxon Mobil',
            'BP': 'BP',
            'PYPL': 'PayPal',
            'SQ': 'Block',
            'COIN': 'Coinbase',
            'RBLX': 'Roblox',
            'FB': 'Facebook',
            'SNAP': 'Snap',
            'TWTR': 'Twitter',
            'PINS': 'Pinterest',
            'SPOT': 'Spotify',
            'U': 'Unity',
            'ZM': 'Zoom',
            'UBER': 'Uber',
            'LYFT': 'Lyft',
            'ABNB': 'Airbnb',
            'DASH': 'DoorDash',
            'GME': 'GameStop',
            'AMC': 'AMC Entertainment',
            'BB': 'BlackBerry'
        }

        company_name = None
        sector = None
        industry = None
        
        try:
            import yfinance as yf
            symbol = symbol.strip().upper()
            print(f"Fetching company information for {symbol}...")

            ticker = yf.Ticker(symbol)
            ticker_info = None

            for attempt in range(3):
                try:
                    ticker_info = ticker.info
                    break
                except Exception as e:
                    if attempt < 2:  # Don't print on last attempt
                        print(f"Attempt {attempt+1} failed: {str(e)}. Retrying...")
                    time.sleep(1)  # Short delay between retries
            
            if not ticker_info:
                raise Exception("Failed to retrieve ticker info after multiple attempts")

            if 'longName' in ticker_info:
                company_name = ticker_info['longName']
            elif 'shortName' in ticker_info:
                company_name = ticker_info['shortName']

            if 'sector' in ticker_info:
                sector = ticker_info['sector']
            if 'industry' in ticker_info:
                industry = ticker_info['industry']

            business_summary = ticker_info.get('longBusinessSummary', '')
            website = ticker_info.get('website', '')

            print(f"Retrieved from yfinance:")
            print(f"- Company name: {company_name}")
            print(f"- Sector: {sector}")
            print(f"- Industry: {industry}")

            if company_name and symbol not in company_map:
                company_map[symbol] = company_name
                
        except Exception as e:
            print(f"Error retrieving data from yfinance: {str(e)}")

        if not company_name:
            company_name = company_map.get(symbol)
            print(f"Using predefined company name from map: {company_name}")

        if not company_name:
            company_name = symbol
            print(f"Using symbol as company name: {symbol}")
        
        print(f"\nGetting news for {company_name} ({symbol})")

        all_headlines = []
        headlines_sources = {}

        print("\nTrying NewsAPI...")
        articles = get_news_newsapi(company_name, days, max_news=50)
        if articles:
            newsapi_headlines = [article['title'] for article in articles]
            print(f"Found {len(newsapi_headlines)} headlines via NewsAPI")
            all_headlines.extend(newsapi_headlines)
            headlines_sources['NewsAPI'] = len(newsapi_headlines)

        print("\nTrying Alpha Vantage...")
        articles = get_news_alphavantage(symbol, company_name)
        if articles:
            av_headlines = [article['title'] for article in articles]
            print(f"Found {len(av_headlines)} headlines via Alpha Vantage")
            all_headlines.extend(av_headlines)
            headlines_sources['Alpha Vantage'] = len(av_headlines)

        if not all_headlines and (sector or industry):
            print("\nTrying to get sector/industry news...")
            
            sector_term = sector if sector else ""
            industry_term = industry if industry else ""
            search_term = f"{sector_term} {industry_term}".strip()
            
            if search_term:
                print(f"Searching for sector news with term: {search_term}")
                sector_articles = get_news_newsapi(search_term, days, max_news=30)
                if sector_articles:
                    sector_headlines = [article['title'] for article in sector_articles]
                    print(f"Found {len(sector_headlines)} sector/industry headlines via NewsAPI")
                    all_headlines.extend(sector_headlines)
                    headlines_sources['Sector NewsAPI'] = len(sector_headlines)

        if not all_headlines:
            print("\nFalling back to general market news...")
            market_terms = ["stock market", "investing", "wall street", "market news"]
            
            for term in market_terms:
                print(f"Searching for general market news with term: {term}")
                market_articles = get_news_newsapi(term, days, max_news=15)
                if market_articles:
                    market_headlines = [article['title'] for article in market_articles]
                    print(f"Found {len(market_headlines)} headlines for '{term}' via NewsAPI")
                    all_headlines.extend(market_headlines)
                    headlines_sources[f'General ({term})'] = len(market_headlines)
                    break

        unique_headlines = []
        seen = set()
        for headline in all_headlines:
            if not headline or len(headline.strip()) == 0:
                continue

            if len(headline) > 200:
                headline = headline[:197] + "..."

            headline_lower = headline.lower()
            if headline_lower not in seen:
                seen.add(headline_lower)
                unique_headlines.append(headline)
        
        if unique_headlines:
            print("\nSummary of headlines found:")
            for source, count in headlines_sources.items():
                print(f"- {source}: {count} headlines")
            print(f"Total unique headlines: {len(unique_headlines)}")

            result = unique_headlines[:max_news]
            print(f"Returning {len(result)} headlines (limit: {max_news})")
            return result

        print("No headlines found from any source")
        return []
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Unexpected error in get_news: {str(e)}")
        return []


def get_full_article(symbol, article_title, days=7):
    try:
        company_map = {
            'AAPL': 'Apple',
            'MSFT': 'Microsoft',
            'GOOGL': 'Google',
            'AMZN': 'Amazon',
            'TSLA': 'Tesla',
            'META': 'Facebook',
            'NFLX': 'Netflix',
            'IBM': 'IBM',
            'INTC': 'Intel',
            'AMD': 'AMD',
            'NVDA': 'NVIDIA',
            'CSCO': 'Cisco',
            'ORCL': 'Oracle',
            'CRM': 'Salesforce',
            'ADBE': 'Adobe'
        }

        company_name = company_map.get(symbol, symbol)

        articles = get_news_newsapi(company_name, days)
        if not articles:
            articles = get_news_alphavantage(symbol, company_name)

        for article in articles:
            if article['title'] == article_title:
                return article['content']

        return None

    except Exception as e:
        print(f"Error retrieving full article: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Update stock data for prediction')
    parser.add_argument('--symbol', type=str, default='AAPL',
                        help='Stock symbol to fetch data for')
    parser.add_argument('--output', type=str, default='latest_stock_data.csv',
                        help='Output file path for updated data')
    parser.add_argument('--days', type=int, default=360,
                        help='Number of days of history to fetch')
    parser.add_argument('--news-output', type=str,
                        help='Output file path for news data')

    args = parser.parse_args()
    update_stock_data(args.symbol, args.output, args.days)
    news = get_news(args.symbol, args.days)
    if news:
        print("\nRecent news that might impact stock price:")
        for i, item in enumerate(news, 1):
            print(f"{i}. {item}")

    print("\nData update complete. You can now run predict.py with the updated data.")