# Инструкции по установке

## Системные требования

- Python 3.8 или выше
- Windows 10/11, macOS, или Linux
- Минимум 8 GB RAM (рекомендуется 16 GB)
- Минимум 5 GB свободного места на диске

## Установка зависимостей

### 1. Создание виртуального окружения (рекомендуется)

```bash
python -m venv venv
```

### 2. Активация виртуального окружения

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### 3. Установка основных зависимостей

```bash
pip install -r requirements.txt
```

### 4. Установка TA-Lib (может потребовать дополнительные шаги)

**Windows:**
```bash
# Скачайте подходящий .whl файл с https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
pip install TA_Lib-0.4.24-cp39-cp39-win_amd64.whl  # замените на ваш файл
```

**macOS:**
```bash
brew install ta-lib
pip install TA-Lib
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install libta-lib-dev
pip install TA-Lib
```

### 5. Настройка NLTK данных

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## Настройка API ключей

Создайте файл `.env` в корневой папке проекта:

```env
# News API
NEWS_API_KEY=your_news_api_key_here
GLOBAL_NEWS_API_KEY=your_global_news_api_key_here

# Alpha Vantage API
ALPHAVANTAGE_KEY=your_alphavantage_key_here

# FRED API (для макроэкономических данных)
FRED_API_KEY=your_fred_api_key_here
```

### Получение API ключей:

1. **News API**: https://newsapi.org/
2. **Alpha Vantage**: https://www.alphavantage.co/support/#api-key
3. **FRED API**: https://fred.stlouisfed.org/docs/api/api_key.html

## Проверка установки

Запустите тест установки:

```bash
python -c "import pandas, numpy, sklearn, torch, transformers, yfinance, matplotlib; print('Все основные библиотеки установлены успешно!')"
```

## Возможные проблемы и решения

### 1. Ошибка установки TA-Lib
- Убедитесь, что у вас установлен Microsoft Visual C++ Build Tools (Windows)
- Используйте предкомпилированные wheel файлы

### 2. Ошибка установки PyTorch
```bash
# Для CPU версии
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Для GPU версии (CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Ошибки с transformers
```bash
pip install --upgrade transformers
```

### 4. Проблемы с памятью
- Увеличьте виртуальную память системы
- Используйте меньшие batch размеры в настройках модели

## Запуск проекта

После успешной установки вы можете запустить:

```bash
# Бэктестинг модели
python backtest.py --tickers AAPL --test_period 30

# Основной скрипт прогнозирования
python .venv/Lib/main.py --ticker AAPL

# Обновление данных
python .venv/Lib/update_data.py --symbol AAPL
```

## Дополнительные рекомендации

1. **Используйте SSD** для лучшей производительности
2. **Регулярно обновляйте зависимости** для получения исправлений безопасности
3. **Мониторьте использование API** чтобы не превысить лимиты
4. **Делайте резервные копии** обученных моделей

## Поддержка

При возникновении проблем:
1. Проверьте версию Python: `python --version`
2. Проверьте установленные пакеты: `pip list`
3. Обновите pip: `pip install --upgrade pip`
4. Переустановите проблемные пакеты 