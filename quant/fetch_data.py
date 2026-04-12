import yfinance as yf
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

tickers = {
    "SPY":    "S&P500 ETF",
    "QQQ":    "Nasdaq100 ETF",
    "AAPL":   "Apple",
    "TSLA":   "Tesla",
    "005930.KS": "Samsung",
    "BTC-USD": "Bitcoin",
}

for symbol, name in tickers.items():
    print(f"Downloading {symbol} ({name})...")
    try:
        df = yf.download(symbol, period="5y", interval="1d", progress=False)
        if df.empty:
            print(f"  SKIP: no data for {symbol}")
            continue

        import pandas as pd
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        df = df.reset_index()
        cols = {}
        for c in df.columns:
            cl = str(c).lower().strip()
            if cl in ('date', 'datetime'): cols[c] = 'Date'
            elif cl == 'open': cols[c] = 'Open'
            elif cl == 'high': cols[c] = 'High'
            elif cl == 'low': cols[c] = 'Low'
            elif cl == 'close': cols[c] = 'Close'
            elif cl == 'volume': cols[c] = 'Volume'
        df = df.rename(columns=cols)

        required = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"  SKIP: missing columns {missing}, available: {list(df.columns)}")
            continue

        df = df[required]
        fname = symbol.replace("-", "").replace(".", "_") + ".csv"
        path = os.path.join(DATA_DIR, fname)
        df.to_csv(path, index=False)
        print(f"  Saved: {path} ({len(df)} rows)")
    except Exception as e:
        print(f"  ERROR: {e}")

print("\nDone.")
