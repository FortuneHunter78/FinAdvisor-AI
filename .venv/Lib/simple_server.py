#!/usr/bin/env python3
"""
Простое веб-приложение для предсказания цен акций
Использует только встроенные библиотеки Python
"""

import json
import sys
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import traceback

# Добавляем текущую директорию к путям Python (где находятся наши модули)
sys.path.insert(0, os.path.dirname(__file__))

try:
    from main import predict_with_models, get_company_info_auto, get_news_headlines, get_global_political_news
    MODULES_LOADED = True
    print("✅ Prediction modules loaded successfully")
except ImportError as e:
    print(f"❌ Warning: Could not import prediction modules: {e}")
    traceback.print_exc()
    MODULES_LOADED = False

class StockPredictionHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/':
            self.serve_html()
        elif parsed_path.path == '/health':
            self.send_json_response({
                "status": "healthy", 
                "modules_loaded": MODULES_LOADED,
                "python_version": sys.version
            })
        elif parsed_path.path == '/api/info':
            self.send_json_response({
                "name": "Stock Prediction API",
                "version": "1.0.0",
                "description": "API для предсказания цен акций",
                "modules_loaded": MODULES_LOADED,
                "python_version": sys.version
            })
        elif parsed_path.path == '/tickers':
            self.handle_tickers()
        else:
            self.send_error(404)
    
    def do_POST(self):
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/predict':
            self.handle_predict()
        else:
            self.send_error(404)
    
    def serve_html(self):
        html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stock Prediction API</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        input, select, button {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
            box-sizing: border-box;
        }
        button {
            background-color: #007bff;
            color: white;
            border: none;
            cursor: pointer;
            margin-top: 10px;
        }
        button:hover {
            background-color: #0056b3;
        }
        button:disabled {
            background-color: #ccc;
            cursor: not-allowed;
        }
        .result {
            margin-top: 20px;
            padding: 15px;
            border-radius: 5px;
            display: none;
        }
        .success {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }
        .error {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
        }
        .loading {
            text-align: center;
            color: #666;
        }
        .status {
            margin-bottom: 20px;
            padding: 10px;
            border-radius: 5px;
            background-color: #e9ecef;
        }
        .checkbox-group {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }
        .checkbox-group input[type="checkbox"] {
            width: auto;
            margin-right: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📈 Stock Prediction API</h1>
        
        <div id="status" class="status">
            <strong>Status:</strong> <span id="statusText">Checking...</span>
        </div>
        
        <form id="predictionForm">
            <div class="form-group">
                <label for="ticker">Stock Ticker:</label>
                <input type="text" id="ticker" name="ticker" placeholder="e.g., AAPL, MSFT, SBER" required>
            </div>
            
            <div class="form-group">
                <label for="days">News Analysis Days:</label>
                <select id="days" name="days">
                    <option value="1">1 day</option>
                    <option value="3" selected>3 days</option>
                    <option value="7">7 days</option>
                </select>
            </div>
            
            <div class="form-group">
                <div class="checkbox-group">
                    <input type="checkbox" id="use_macro" name="use_macro" checked>
                    <label for="use_macro">Use Macroeconomic Analysis</label>
                </div>
            </div>
            
            <div class="form-group">
                <div class="checkbox-group">
                    <input type="checkbox" id="is_moex" name="is_moex">
                    <label for="is_moex">Russian Stock (MOEX)</label>
                </div>
            </div>
            
            <button type="submit" id="predictBtn">🔮 Predict Stock Price</button>
        </form>
        
        <div id="result" class="result">
            <div id="resultContent"></div>
        </div>
    </div>

    <script>
        // Проверяем статус при загрузке
        fetch('/health')
            .then(response => response.json())
            .then(data => {
                const statusText = document.getElementById('statusText');
                if (data.modules_loaded) {
                    statusText.innerHTML = '✅ Ready for predictions';
                    statusText.style.color = 'green';
                } else {
                    statusText.innerHTML = '❌ Prediction modules not available';
                    statusText.style.color = 'red';
                    document.getElementById('predictBtn').disabled = true;
                }
            })
            .catch(error => {
                document.getElementById('statusText').innerHTML = '❌ Server error';
                document.getElementById('statusText').style.color = 'red';
            });

        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(e.target);
            const data = {
                ticker: formData.get('ticker').toUpperCase(),
                days: parseInt(formData.get('days')),
                use_macro: document.getElementById('use_macro').checked,
                is_moex: document.getElementById('is_moex').checked
            };
            
            const resultDiv = document.getElementById('result');
            const resultContent = document.getElementById('resultContent');
            const predictBtn = document.getElementById('predictBtn');
            
            // Show loading
            resultDiv.className = 'result loading';
            resultDiv.style.display = 'block';
            resultContent.innerHTML = '⏳ Analyzing stock data and making prediction...';
            predictBtn.disabled = true;
            predictBtn.textContent = 'Processing...';
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    resultDiv.className = 'result success';
                    const direction = result.direction > 0 ? '📈 UP' : result.direction < 0 ? '📉 DOWN' : '➡️ NEUTRAL';
                    const changeColor = result.change_percent > 0 ? 'green' : result.change_percent < 0 ? 'red' : 'gray';
                    
                    resultContent.innerHTML = `
                        <h3>🎯 Prediction Results for ${result.ticker}</h3>
                        <p><strong>Current Price:</strong> $${result.current_price.toFixed(2)}</p>
                        <p><strong>Predicted Price:</strong> $${result.predicted_price.toFixed(2)}</p>
                        <p><strong>Direction:</strong> ${direction}</p>
                        <p><strong>Change:</strong> <span style="color: ${changeColor}">${result.change_percent > 0 ? '+' : ''}${result.change_percent.toFixed(2)}%</span></p>
                        <p><strong>News Analyzed:</strong> ${result.news_count} articles</p>
                        <p><strong>Sentiment:</strong> ${result.sentiment}</p>
                    `;
                } else {
                    throw new Error(result.error || 'Prediction failed');
                }
            } catch (error) {
                resultDiv.className = 'result error';
                resultContent.innerHTML = `
                    <h3>❌ Error</h3>
                    <p>${error.message}</p>
                `;
            } finally {
                predictBtn.disabled = false;
                predictBtn.textContent = '🔮 Predict Stock Price';
            }
        });
    </script>
</body>
</html>
        """
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.send_header('Content-length', len(html_content.encode('utf-8')))
        self.end_headers()
        self.wfile.write(html_content.encode('utf-8'))
    
    def send_json_response(self, data, status_code=200):
        response = json.dumps(data, ensure_ascii=False, indent=2)
        
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json; charset=utf-8')
        self.send_header('Content-length', len(response.encode('utf-8')))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(response.encode('utf-8'))
    
    def handle_tickers(self):
        if not MODULES_LOADED:
            self.send_json_response({"error": "Prediction modules not available"}, 500)
            return
        
        try:
            from main import MOEX_TICKERS, SECTORS
            
            moex_tickers = list(MOEX_TICKERS.keys())
            us_tickers = [ticker for ticker in SECTORS.keys() if ticker not in MOEX_TICKERS]
            
            response = {
                "moex_tickers": moex_tickers,
                "us_tickers": us_tickers,
                "total_count": len(SECTORS)
            }
            self.send_json_response(response)
        except Exception as e:
            self.send_json_response({"error": str(e)}, 500)
    
    def handle_predict(self):
        if not MODULES_LOADED:
            self.send_json_response({"error": "Prediction modules not available"}, 500)
            return
        
        try:
            # Читаем данные запроса
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))
            
            # Извлекаем параметры
            ticker = request_data.get('ticker', '').upper()
            days = request_data.get('days', 3)
            use_macro = request_data.get('use_macro', True)
            is_moex = request_data.get('is_moex', False)
            
            if not ticker:
                self.send_json_response({"error": "Ticker is required"}, 400)
                return
            
            print(f"🔍 Processing prediction for {ticker}")
            
            # Получаем информацию о компании
            company_info = get_company_info_auto(ticker, is_moex)
            
            # Получаем новости
            news_titles = get_news_headlines(
                ticker, 
                days=days, 
                max_news=100, 
                is_moex=is_moex
            )
            
            # Получаем глобальные новости
            global_news = get_global_political_news(
                days=days, 
                max_news=20, 
                company_info=company_info
            )
            
            # Делаем предсказание
            results = predict_with_models(
                ticker=ticker,
                model_base_path="models",
                all_news=news_titles,
                use_macro=use_macro,
                is_moex=is_moex,
                company_info=company_info,
                global_news=global_news
            )
            
            if not results or "error" in results:
                error_msg = results.get('error', 'Unknown error') if results else 'No results'
                self.send_json_response({"error": f"Prediction failed: {error_msg}"}, 500)
                return
            
            # Формируем ответ
            current_price = results.get('current_price', 0)
            predicted_price = results.get('combined_prediction', {}).get('price', current_price)
            direction = results.get('combined_prediction', {}).get('direction', 0)
            change_percent = ((predicted_price / current_price - 1) * 100) if current_price > 0 else 0
            
            news_analysis = results.get('news_analysis', {})
            news_count = news_analysis.get('news_count', 0)
            
            # Определяем доминирующий сентимент
            sentiment_percentages = news_analysis.get('sentiment_percentages', {})
            sentiment = "neutral"
            if sentiment_percentages:
                max_sentiment = max(sentiment_percentages.items(), key=lambda x: x[1])
                sentiment = max_sentiment[0]
            
            response = {
                "ticker": ticker,
                "current_price": current_price,
                "predicted_price": predicted_price,
                "direction": direction,
                "change_percent": change_percent,
                "news_count": news_count,
                "sentiment": sentiment
            }
            
            print(f"✅ Prediction completed for {ticker}")
            self.send_json_response(response)
            
        except json.JSONDecodeError:
            self.send_json_response({"error": "Invalid JSON"}, 400)
        except Exception as e:
            print(f"❌ Error in prediction: {e}")
            traceback.print_exc()
            self.send_json_response({"error": str(e)}, 500)
    
    def log_message(self, format, *args):
        # Переопределяем для более чистого вывода
        print(f"{self.address_string()} - {format % args}")

def run_server(port=8000):
    server_address = ('', port)
    httpd = HTTPServer(server_address, StockPredictionHandler)
    
    print(f"🚀 Stock Prediction Server starting on port {port}")
    print(f"📊 Modules loaded: {MODULES_LOADED}")
    print(f"🐍 Python version: {sys.version}")
    print(f"🌐 Open http://localhost:{port} in your browser")
    print("Press Ctrl+C to stop the server")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
        httpd.server_close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Stock Prediction Web Server')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the server on')
    args = parser.parse_args()
    
    run_server(args.port) 