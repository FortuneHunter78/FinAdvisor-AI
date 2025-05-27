import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import os
from fredapi import Fred
import yfinance as yf
import time

class MacroDataLoader:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.fred = None
        if api_key:
            try:
                self.fred = Fred(api_key=api_key)
                print("FRED API успешно инициализирован")
            except Exception as e:
                print(f"Ошибка при инициализации FRED API: {str(e)}")
    
    def get_fred_data(self, indicators, start_date=None, end_date=None):
        if not self.fred:
            print("FRED API не инициализирован. Проверьте API ключ.")
            return None
        
        if not start_date:
            start_date = datetime.now() - timedelta(days=5*365)
        if not end_date:
            end_date = datetime.now()
            
        result = pd.DataFrame()
        
        try:
            for name, series_id in indicators.items():
                print(f"Загрузка данных для {name} (series_id: {series_id})")
                series = self.fred.get_series(series_id, start_date, end_date)
                if series is not None and not series.empty:
                    temp_df = pd.DataFrame({name: series})
                    if result.empty:
                        result = temp_df
                    else:
                        result = result.join(temp_df, how='outer')
                else:
                    print(f"Не удалось получить данные для {name}")

            result = result.fillna(method='ffill')
            
            print(f"Загружено макроэкономических показателей: {len(result.columns)}")
            print(f"Временной диапазон: {result.index.min()} - {result.index.max()}")
            
            return result
        
        except Exception as e:
            print(f"Ошибка при загрузке данных из FRED: {str(e)}")
            return None
    
    def get_alternative_macro_data(self, start_date=None, end_date=None):
        if not start_date:
            start_date = datetime.now() - timedelta(days=5*365)
        if not end_date:
            end_date = datetime.now()

        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        macro_tickers = {
            'Treasury_10Y': '^TNX',      # 10-Year Treasury Rate
            'Treasury_5Y': '^FVX',       # 5-Year Treasury Rate
            'Treasury_1Y': '^IRX',       # 1-Year Treasury Rate
            'VIX': '^VIX',               # Volatility Index
            'USD_Index': 'DX-Y.NYB',     # US Dollar Index
            'Gold': 'GC=F',              # Gold Futures
            'Oil_WTI': 'CL=F',           # Crude Oil WTI Futures
            'S&P500': '^GSPC',           # S&P 500
            'Nasdaq': '^IXIC',           # Nasdaq Composite
            'DowJones': '^DJI',          # Dow Jones Industrial Average
            'Russell2000': '^RUT',       # Russell 2000 (Small-Cap)
        }
        
        macro_data = pd.DataFrame()
        
        for name, ticker in macro_tickers.items():
            try:
                print(f"Загрузка данных для {name} (ticker: {ticker})")
                time.sleep(0.5)

                data = yf.download(ticker, start=start_str, end=end_str, progress=False)
                
                if not data.empty:
                    close_series = data['Close']
                    close_series.name = name
                    
                    if macro_data.empty:
                        macro_data = pd.DataFrame(close_series)
                    else:
                        macro_data = macro_data.join(close_series, how='outer')
                else:
                    print(f"Нет данных для {ticker}")
            
            except Exception as e:
                print(f"Ошибка при загрузке {ticker}: {str(e)}")
                continue

        macro_data = macro_data.fillna(method='ffill')

        if 'Treasury_10Y' in macro_data.columns and 'Treasury_1Y' in macro_data.columns:
            macro_data['Yield_Curve'] = macro_data['Treasury_10Y'] - macro_data['Treasury_1Y']
        
        if 'S&P500' in macro_data.columns:
            macro_data['S&P500_Return'] = macro_data['S&P500'].pct_change()
            macro_data['S&P500_Volatility'] = macro_data['S&P500_Return'].rolling(window=20).std()
        
        print(f"Загружено макроэкономических показателей: {len(macro_data.columns)}")
        print(f"Временной диапазон: {macro_data.index.min()} - {macro_data.index.max()}")
        
        return macro_data
    
    def get_economic_calendar_data(self, days=30):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        calendar_data = pd.DataFrame(index=date_range)
        calendar_data['FOMC_Meeting'] = np.random.choice([0, 1], size=len(date_range), p=[0.95, 0.05])
        calendar_data['GDP_Release'] = np.random.choice([0, 1], size=len(date_range), p=[0.93, 0.07])
        calendar_data['Unemployment_Release'] = np.random.choice([0, 1], size=len(date_range), p=[0.93, 0.07])
        calendar_data['CPI_Release'] = np.random.choice([0, 1], size=len(date_range), p=[0.93, 0.07])
        calendar_data['FedRate_Change'] = np.random.choice([0, 1], size=len(date_range), p=[0.98, 0.02])
        
        print(f"Создан демонстрационный экономический календарь на {len(calendar_data)} дней")
        
        return calendar_data
    
    def merge_macro_with_stock_data(self, stock_data, macro_data):
        stock_df = stock_data.copy()
        if 'Date' in stock_df.columns:
            stock_df['Date'] = pd.to_datetime(stock_df['Date'])
            stock_df = stock_df.set_index('Date')

        merged_df = stock_df.join(macro_data, how='left')
        merged_df = merged_df.fillna(method='ffill')
        merged_df = merged_df.reset_index()
        
        print(f"Объединено {len(stock_df.columns)} столбцов из данных акций и "
              f"{len(macro_data.columns)} макроэкономических показателей")
        
        return merged_df
    
    def get_all_macro_features(self, days=360, country=None):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        print(f"Получаем макроэкономические данные за последние {days} дней...")
        if country:
            print(f"Учитываем особенности страны: {country}")

        alt_data = self.get_alternative_macro_data(start_date, end_date)
        fred_data = None
        if self.fred:
            indicators = {
                'GDP': 'GDP',                  # Gross Domestic Product
                'Unemployment': 'UNRATE',      # Unemployment Rate
                'CPI': 'CPIAUCSL',             # Consumer Price Index
                'FedRate': 'FEDFUNDS',         # Federal Funds Rate
                'M2': 'M2SL',                  # M2 Money Supply
                'Industrial_Production': 'INDPRO',  # Industrial Production Index
                'Retail_Sales': 'RSAFS',       # Retail Sales
                'Housing_Starts': 'HOUST',     # Housing Starts
                'Consumer_Sentiment': 'UMCSENT',  # Consumer Sentiment Index
                'PCE': 'PCE',                  # Personal Consumption Expenditures
            }

            if country:
                if country == 'usa':
                    indicators.update({
                        'ISM_Manufacturing': 'NAPM',       # ISM Manufacturing Index
                        'Building_Permits': 'PERMIT',      # Building Permits
                        'Nonfarm_Payrolls': 'PAYEMS',      # Nonfarm Payrolls
                        'Durable_Goods': 'DGORDER',        # Durable Goods Orders
                        'Trade_Balance': 'BOPGSTB',        # Trade Balance
                    })
                elif country == 'russia':
                    pass
                elif country in ['germany', 'france', 'eu']:
                    indicators.update({
                        'EU_Unemployment': 'LRHUTTTTEZM156S',  # EU Unemployment Rate
                        'EU_CPI': 'CP0000EZ19M086NEST',        # EU Consumer Price Index
                        'ECB_Rate': 'ECBDFR',                  # ECB Deposit Facility Rate
                    })
                elif country == 'china':
                    pass
                elif country == 'japan':
                    indicators.update({
                        'Japan_GDP': 'JPNNGDP',            # Japan GDP
                        'Japan_CPI': 'JPNCPIALLMINMEI',    # Japan CPI
                        'BOJ_Rate': 'IRSTCB01JPM156N',     # Bank of Japan Interest Rate
                    })
            
            fred_data = self.get_fred_data(indicators, start_date, end_date)
        if fred_data is not None and not fred_data.empty:
            if alt_data is not None and not alt_data.empty:
                macro_data = alt_data.join(fred_data, how='outer')
            else:
                macro_data = fred_data
        else:
            macro_data = alt_data

        calendar_data = self.get_economic_calendar_data(days)
        
        if calendar_data is not None and not calendar_data.empty:
            if macro_data is not None and not macro_data.empty:
                macro_data = macro_data.join(calendar_data, how='outer')
            else:
                macro_data = calendar_data

        if macro_data is not None:
            macro_data = macro_data.fillna(method='ffill').fillna(method='bfill').fillna(0)
            for col in macro_data.select_dtypes(include=['float64', 'int64']).columns:
                if not ('Return' in col or 'Change' in col or 'Volatility' in col or 
                        'FOMC' in col or 'Release' in col or 'Yield' in col):
                    macro_data[f'{col}_Change'] = macro_data[col].pct_change()

            print("Рассчитываем изменения макроэкономических показателей за последние 7 дней...")
            last_7_days = macro_data.tail(7) if len(macro_data) >= 7 else macro_data
            
            for col in last_7_days.select_dtypes(include=['float64', 'int64']).columns:
                if not ('Return' in col or 'Change' in col or 'Volatility' in col or 
                        'FOMC' in col or 'Release' in col or 'Yield' in col):
                    try:
                        first_val = last_7_days[col].iloc[0]
                        last_val = last_7_days[col].iloc[-1]
                        
                        if first_val != 0:
                            change_pct = ((last_val - first_val) / abs(first_val)) * 100
                            macro_data[f'{col}_7d_Change'] = change_pct

                            if abs(change_pct) > 1.0:
                                direction = "🔺" if change_pct > 0 else "🔻"
                                print(f"  {direction} {col}: изменение за 7 дней {change_pct:.2f}%")
                    except Exception as e:
                        pass
            
            print(f"Сформирован набор макроэкономических показателей из {len(macro_data.columns)} столбцов")
        else:
            print("Не удалось получить макроэкономические данные")
        
        return macro_data

    def get_key_macro_indicators(self, days=365, country=None):
        print(f"Анализ ключевых макроэкономических показателей за последние {days} дней...")

        if country == 'russia':
            indicators = {
                'GDP': 'RUSRGDPEXP',    # Russian Real GDP
                'CPI': 'RUSCPIALLMINMEI',  # Russian CPI
                'Unemployment': 'LMUNRRTTRUSM156S'  # Russian Unemployment Rate
            }
        else:
            indicators = {
                'GDP': 'GDP',           # US GDP
                'CPI': 'CPIAUCSL',      # US Consumer Price Index
                'Unemployment': 'UNRATE' # US Unemployment Rate
            }

        data = None
        if self.fred:
            try:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                data = self.get_fred_data(indicators, start_date, end_date)
            except Exception as e:
                print(f"Ошибка при получении данных из FRED: {str(e)}")

        if data is None or data.empty:
            print("Используем альтернативные источники данных...")
            data = pd.DataFrame()
            
            try:
                start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
                end_date = datetime.now().strftime('%Y-%m-%d')

                if country != 'russia':
                    gdp_data = yf.download('SPY', start=start_date, end=end_date)
                    if not gdp_data.empty:
                        data.loc[:, 'GDP'] = gdp_data['Close']
                    tip_data = yf.download('TIP', start=start_date, end=end_date)
                    ief_data = yf.download('IEF', start=start_date, end=end_date)
                    if not tip_data.empty and not ief_data.empty:
                        data.loc[:, 'CPI'] = tip_data['Close'].values / ief_data['Close'].values

                    vix_data = yf.download('^VIX', start=start_date, end=end_date)
                    if not vix_data.empty:
                        data.loc[:, 'Unemployment'] = vix_data['Close']
                else:
                    moex_data = yf.download('IMOEX.ME', start=start_date, end=end_date)
                    if not moex_data.empty:
                        data.loc[:, 'GDP'] = moex_data['Close']

                    usdrub_data = yf.download('RUB=X', start=start_date, end=end_date)
                    if not usdrub_data.empty:
                        data.loc[:, 'CPI'] = usdrub_data['Close']

                    rtsi_data = yf.download('RTSI.ME', start=start_date, end=end_date)
                    if not rtsi_data.empty:
                        data.loc[:, 'Unemployment'] = 10000 / rtsi_data['Close'].values

                if not data.empty:
                    data.index = gdp_data.index if country != 'russia' else moex_data.index
                
            except Exception as e:
                print(f"Ошибка при получении альтернативных данных: {str(e)}")

        trends = {}
        if not data.empty:
            print("\n📊 АНАЛИЗ МАКРОЭКОНОМИЧЕСКИХ ПОКАЗАТЕЛЕЙ:")
            
            for indicator in ['GDP', 'CPI', 'Unemployment']:
                if indicator in data.columns:
                    series = data[indicator].dropna()
                    if len(series) > 0:
                        start_value = series.iloc[0]
                        end_value = series.iloc[-1]
                        if start_value != 0:
                            change_pct = ((end_value - start_value) / abs(start_value)) * 100

                            if abs(change_pct) < 1.0:
                                trend = 0
                                trend_icon = "➡️"
                                trend_text = "стабилен"
                            else:
                                if change_pct > 0:
                                    trend = 1
                                    trend_icon = "🔺"
                                    trend_text = "растет"
                                else:
                                    trend = -1
                                    trend_icon = "🔻"
                                    trend_text = "падает"
                            
                            trends[indicator] = {
                                'trend': trend,
                                'change_percent': change_pct,
                                'start_value': float(start_value),
                                'end_value': float(end_value),
                                'last_value_date': data.index[-1].strftime('%Y-%m-%d')
                            }

                            indicator_names = {
                                'GDP': 'ВВП',
                                'CPI': 'Инфляция',
                                'Unemployment': 'Безработица'
                            }
                            
                            print(f"\n{trend_icon} {indicator_names.get(indicator, indicator)}:")
                            print(f"  • Тренд: {trend_text}")
                            print(f"  • Изменение: {'+' if change_pct > 0 else ''}{change_pct:.2f}%")
                            print(f"  • Начальное значение: {start_value:.2f}")
                            print(f"  • Конечное значение: {end_value:.2f}")
                            print(f"  • Последнее обновление: {data.index[-1].strftime('%Y-%m-%d')}")

                            if indicator == 'GDP':
                                if trend == 1:
                                    print("  • Интерпретация: Экономический рост 📈")
                                elif trend == -1:
                                    print("  • Интерпретация: Экономический спад 📉")
                                else:
                                    print("  • Интерпретация: Стабильная экономика ⚖️")
                            elif indicator == 'CPI':
                                if trend == 1:
                                    print("  • Интерпретация: Рост инфляционного давления 💹")
                                elif trend == -1:
                                    print("  • Интерпретация: Снижение инфляции 📊")
                                else:
                                    print("  • Интерпретация: Стабильный уровень цен ⚖️")
                            elif indicator == 'Unemployment':
                                if trend == 1:
                                    print("  • Интерпретация: Ухудшение ситуации на рынке труда 📊")
                                elif trend == -1:
                                    print("  • Интерпретация: Улучшение ситуации на рынке труда 📈")
                                else:
                                    print("  • Интерпретация: Стабильный рынок труда ⚖️")
        
        return trends

class MacroAdjustment:
    def __init__(self):
        self.sector_weights = {
            'energy': {
                'Oil_WTI': 0.8,
                'Treasury_10Y': 0.3,
                'USD_Index': 0.5,
                'S&P500': 0.4,
                'VIX': 0.5
            },
            'technology': {
                'Nasdaq': 0.8,
                'Treasury_10Y': 0.6,
                'VIX': 0.7,
                'S&P500': 0.5,
            },
            'finance': {
                'Treasury_10Y': 0.8,
                'Treasury_1Y': 0.6,
                'FedRate': 0.9,
                'S&P500': 0.6,
                'Yield_Curve': 0.7,
                'VIX': 0.7
            },
            'materials': {
                'Gold': 0.7,
                'USD_Index': 0.6,
                'S&P500': 0.4,
                'Oil_WTI': 0.5
            },
            'consumer': {
                'Retail_Sales': 0.8,
                'Consumer_Sentiment': 0.7,
                'S&P500': 0.5,
                'Treasury_10Y': 0.4
            },
            'healthcare': {
                'S&P500': 0.5,
                'Treasury_10Y': 0.4,
                'VIX': 0.4
            },
            'utilities': {
                'Treasury_10Y': 0.8,
                'S&P500': 0.3,
                'FedRate': 0.7
            },
            'transportation': {
                'Oil_WTI': 0.7,
                'Industrial_Production': 0.6,
                'S&P500': 0.5
            },
            'automotive': {
                'Consumer_Sentiment': 0.7,
                'Industrial_Production': 0.6,
                'Oil_WTI': 0.5,
                'S&P500': 0.5
            },
            'communication': {
                'S&P500': 0.6,
                'Nasdaq': 0.7,
                'Treasury_10Y': 0.4,
                'VIX': 0.5
            },
            'real_estate': {
                'Treasury_10Y': 0.9,
                'Housing_Starts': 0.8,
                'FedRate': 0.8
            },
            'industrial': {
                'Industrial_Production': 0.8,
                'S&P500': 0.5,
                'Oil_WTI': 0.5
            }
        }

        self.country_weights = {
            'usa': {
                'S&P500': 0.9,
                'Treasury_10Y': 0.8,
                'FedRate': 0.8,
                'Unemployment': 0.7,
                'CPI': 0.7,
                'GDP': 0.6
            },
            'china': {
                'Shanghai': 0.8,
                'USD_Index': 0.6,
                'Oil_WTI': 0.5
            },
            'russia': {
                'Oil_WTI': 0.9,
                'Gold': 0.7,
                'USD_Index': 0.8
            },
            'eu': {
                'EU_Unemployment': 0.7,
                'EU_CPI': 0.7,
                'ECB_Rate': 0.8,
                'EUR/USD': 0.6
            },
            'uk': {
                'FTSE': 0.8,
                'Oil_WTI': 0.5,
                'USD_Index': 0.6
            },
            'japan': {
                'Nikkei': 0.8,
                'USD_Index': 0.6,
                'Japan_GDP': 0.7,
                'Japan_CPI': 0.6,
                'BOJ_Rate': 0.7
            }
        }
    
    def _get_metric_impact(self, metric, value, direction):
        impact = abs(value) / 10.0
        impact = min(impact, 0.5)

        positive_correlation = ['S&P500', 'Nasdaq', 'DowJones', 'Russell2000',
                              'GDP', 'Industrial_Production', 'Retail_Sales', 
                              'Consumer_Sentiment', 'Housing_Starts', 'Building_Permits',
                              'Yield_Curve', 'FTSE', 'Nikkei', 'Shanghai']
                              
        negative_correlation = ['Treasury_10Y', 'Treasury_5Y', 'Treasury_1Y', 
                              'VIX', 'Unemployment', 'CPI', 'FedRate', 'interest_rate',
                              'ECB_Rate', 'BOJ_Rate']

        special_cases = ['Oil_WTI', 'USD_Index', 'Gold', 'EUR/USD']

        if metric in positive_correlation:
            sign = 1 if value > 0 else -1
        elif metric in negative_correlation:
            sign = -1 if value > 0 else 1
        elif metric in special_cases:
            sign = 0
        else:
            return 0.0

        if direction == 'down':
            sign = -sign
            
        return sign * impact
    
    def _analyze_interest_rate_impact(self, macro_trends, sector):
        rate_indicators = ['FedRate', 'interest_rate', 'Treasury_10Y', 'Treasury_1Y']
        rate_data = None
        rate_indicator = None
        
        for indicator in rate_indicators:
            if indicator in macro_trends:
                rate_data = macro_trends[indicator]
                rate_indicator = indicator
                break
        
        if not rate_data:
            return {
                'has_impact': False,
                'impact': 0.0,
                'message': "Данные о процентной ставке отсутствуют"
            }

        if isinstance(rate_data, dict) and 'change_percent' in rate_data:
            change_percent = rate_data.get('change_percent', 0)
        else:
            change_percent = 0

        rate_sensitive_sectors = {
            'finance': 0.9,
            'real_estate': -0.8,
            'utilities': -0.7,
            'technology': -0.6,
            'consumer': -0.5,
            'energy': -0.3,
            'healthcare': -0.4,
            'materials': -0.4,
        }

        sensitivity = rate_sensitive_sectors.get(sector, -0.5)
        raw_impact = sensitivity * change_percent / 100
        impact = max(min(raw_impact, 0.01), -0.01)
        if abs(impact) < 0.001:
            message = "Изменение процентной ставки оказывает минимальное влияние"
        else:
            direction = "повышение" if change_percent > 0 else "снижение"
            effect = "положительно" if impact > 0 else "отрицательно"
            message = f"{direction} процентной ставки {effect} влияет на сектор {sector}"
        
        return {
            'has_impact': abs(impact) >= 0.001,
            'impact': impact,
            'indicator': rate_indicator,
            'change_percent': change_percent,
            'sensitivity': sensitivity,
            'message': message
        }
    
    def _get_sector_impact(self, sector, macro_changes):
        impact = 0.0
        count = 0

        if sector in self.sector_weights:
            sector_weights = self.sector_weights[sector]
            for metric, weight in sector_weights.items():
                metric_key = None
                if metric in macro_changes:
                    metric_key = metric
                elif f"{metric}_7d_Change" in macro_changes:
                    metric_key = f"{metric}_7d_Change"
                elif f"{metric}_Change" in macro_changes:
                    metric_key = f"{metric}_Change"

                if metric_key and abs(macro_changes[metric_key]) > 0.5:
                    if metric == 'Oil_WTI':
                        if sector in ['energy']:
                            impact += weight * (macro_changes[metric_key] / 10.0)
                        elif sector in ['transportation', 'automotive', 'consumer']:
                            impact -= weight * (macro_changes[metric_key] / 10.0)
                    elif metric == 'USD_Index':
                        if sector in ['technology', 'consumer'] and sector != 'energy':
                            impact -= weight * (macro_changes[metric_key] / 10.0)
                        elif sector in ['energy', 'materials']:
                            impact -= weight * (macro_changes[metric_key] / 10.0)
                    else:
                        impact += weight * self._get_metric_impact(metric, macro_changes[metric_key], 'up')
                    
                    count += 1

        if count == 0:
            return 0.0

        return impact / max(count, 1)
    
    def _get_country_impact(self, country, macro_changes):
        impact = 0.0
        count = 0

        if country in self.country_weights:
            country_weights = self.country_weights[country]
            for metric, weight in country_weights.items():
                metric_key = None
                if metric in macro_changes:
                    metric_key = metric
                elif f"{metric}_7d_Change" in macro_changes:
                    metric_key = f"{metric}_7d_Change"
                elif f"{metric}_Change" in macro_changes:
                    metric_key = f"{metric}_Change"
                if metric_key and abs(macro_changes[metric_key]) > 0.5:
                    impact += weight * self._get_metric_impact(metric, macro_changes[metric_key], 'up')
                    count += 1

        if count == 0:
            return 0.0

        return impact / max(count, 1)
    
    def adjust_prediction(self, base_prediction, company_info=None):
        if not company_info or 'macro_trends' not in company_info:
            return base_prediction
        try:
            base_price = float(base_prediction['price'])
        except (KeyError, TypeError, ValueError):
            print("Warning: Invalid base price in prediction")
            return base_prediction

        adjusted_prediction = base_prediction.copy()

        macro_trends = company_info['macro_trends']

        weights = {
            'GDP': 0.4,
            'CPI': 0.35,
            'Unemployment': 0.25
        }

        total_impact = 0.0
        sector_impact = 0.0
        country_impact = 0.0

        sector = company_info.get('sector', 'unknown')
        country = company_info.get('country', 'unknown')

        for indicator, trend_info in macro_trends.items():
            if isinstance(trend_info, dict) and 'trend' in trend_info:
                try:
                    trend = float(trend_info['trend'])
                    change_percent = float(trend_info.get('change_percent', 0))
                    change_percent = max(min(change_percent, 1.0), -1.0)
                    metric_impact = self._get_metric_impact(indicator, change_percent, trend)
                    weighted_impact = float(metric_impact) * weights.get(indicator, 0.0)
                    total_impact += weighted_impact
                except (TypeError, ValueError) as e:
                    print(f"Warning: Error processing {indicator} impact: {e}")
                    continue

        if sector != 'unknown':
            try:
                sector_impact = float(self._get_sector_impact(sector, macro_trends))
            except (TypeError, ValueError) as e:
                print(f"Warning: Error calculating sector impact: {e}")
                sector_impact = 0.0
                
        if country != 'unknown':
            try:
                country_impact = float(self._get_country_impact(country, macro_trends))
            except (TypeError, ValueError) as e:
                print(f"Warning: Error calculating country impact: {e}")
                country_impact = 0.0

        total_impact = max(min(float(total_impact), 0.01), -0.01)
        sector_impact = max(min(float(sector_impact), 0.01), -0.01)
        country_impact = max(min(float(country_impact), 0.01), -0.01)

        interest_rate_analysis = self._analyze_interest_rate_impact(macro_trends, sector)
        interest_rate_impact = interest_rate_analysis.get('impact', 0.0)

        if interest_rate_analysis.get('has_impact', False):
            print(f"\n💰 АНАЛИЗ ПРОЦЕНТНОЙ СТАВКИ:")
            print(f"  • Показатель: {interest_rate_analysis.get('indicator', 'unknown')}")
            print(f"  • Изменение: {interest_rate_analysis.get('change_percent', 0):.2f}%")
            print(f"  • Чувствительность сектора {sector}: {interest_rate_analysis.get('sensitivity', 0):.2f}")
            print(f"  • Влияние на прогноз: {interest_rate_impact * 100:.2f}%")
            print(f"  • {interest_rate_analysis.get('message', '')}")

        final_impact = (total_impact + sector_impact + country_impact + interest_rate_impact) / 4.0
        final_impact = max(min(final_impact, 0.01), -0.01)
        price_adjustment = base_price * final_impact
        adjusted_prediction['price'] = base_price + price_adjustment
        current_direction = base_prediction.get('direction', 0)
        try:
            current_direction = int(current_direction)
        except (TypeError, ValueError):
            current_direction = 0
            
        if final_impact > 0:
            new_direction = 1
        elif final_impact < 0:
            new_direction = -1
        else:
            new_direction = 0
            
        adjusted_prediction['direction'] = new_direction
        adjusted_prediction['macro_impact'] = {
            'total_impact': float(total_impact),
            'sector_impact': float(sector_impact),
            'country_impact': float(country_impact),
            'interest_rate_impact': float(interest_rate_impact),
            'final_impact': float(final_impact),
            'price_adjustment': float(price_adjustment),
            'direction_changed': new_direction != current_direction
        }
        interest_rate_impact = self._analyze_interest_rate_impact(macro_trends, sector)
        adjusted_prediction['interest_rate_impact'] = interest_rate_impact
        
        return adjusted_prediction

if __name__ == "__main__":
    loader = MacroDataLoader()
    macro_data = loader.get_all_macro_features(days=365)
    
    if macro_data is not None:
        macro_data.to_csv("macro_data.csv")
        print(f"Данные сохранены в macro_data.csv")
        print(f"Количество записей: {len(macro_data)}")
        print(f"Доступные показатели: {macro_data.columns.tolist()}") 