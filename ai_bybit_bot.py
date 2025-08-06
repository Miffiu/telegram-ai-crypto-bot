import ccxt
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
import requests
import logging

# === KONFIGURACJA ===
TELEGRAM_BOT_TOKEN = "8477709111:AAFOkcyGSw_oKmgD6ShuBZYKG92ryadl_iI"
TELEGRAM_CHAT_ID = "5352662762"
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
TIMEFRAME = "5m"
CHECK_INTERVAL = 60  # co ile sekund bot działa

exchange = ccxt.bybit({
    'apiKey': 'fuZqZ8RCUh9byXQFCW',
    'secret': 'KvTLGz1fpSG4wHQ1oyvFYifuelqDdqprsfo7',
    'enableRateLimit': True,
    'options': {
        'defaultType': 'spot',
    },
})

# === WYSYŁANIE WIADOMOŚCI DO TELEGRAMA (requests) ===
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }
    try:
        response = requests.post(url, data=payload)
        if response.status_code != 200:
            logging.error(f"Błąd Telegrama: {response.text}")
    except Exception as e:
        logging.error(f"Wyjątek przy wysyłaniu wiadomości: {e}")

# === MODEL AI (prosty RandomForest na świecach) ===

def prepare_features(df):
    df['return'] = df['close'].pct_change()
    df['volatility'] = df['return'].rolling(5).std()
    df['sma'] = df['close'].rolling(5).mean()
    df = df.dropna()
    X = df[['return', 'volatility', 'sma']]
    y = np.where(df['close'].shift(-1) > df['close'], 1, 0)  # 1 = BUY, 0 = SELL
    return X, y

def train_model(df):
    X, y = prepare_features(df)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    return model

def predict_signal(model, df):
    X, _ = prepare_features(df)
    if X.empty:
        return "HOLD"
    pred = model.predict(X.tail(1))
    return "BUY" if pred[0] == 1 else "SELL"

# === POBIERANIE DANYCH ===

def fetch_ohlcv(symbol):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=100)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    return df

# === GŁÓWNA PĘTLA ===

def main():
    logging.basicConfig(level=logging.INFO)
    models = {}  # osobny model dla każdej pary

    send_telegram_message("🤖 Bot AI połączony i gotowy!")

    while True:
        for symbol in SYMBOLS:
            try:
                df = fetch_ohlcv(symbol)
                if df.empty:
                    continue

                if symbol not in models:
                    models[symbol] = train_model(df)
                    logging.info(f"🔧 Model wytrenowany dla {symbol}")

                signal = predict_signal(models[symbol], df)
                message = f"📊 Sygnał AI dla {symbol} (TF: {TIMEFRAME}): {signal}"
                logging.info(message)
                send_telegram_message(message)

            except Exception as e:
                logging.error(f"Błąd dla {symbol}: {str(e)}")
        time.sleep(CHECK_INTERVAL)

if __name__ == '__main__':
    main()