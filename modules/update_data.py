import os
import time
import pandas as pd
import yfinance as yf
import requests

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

TICKERS = [
    "PETR4.SA",
    "VALE3.SA",
    "ITUB4.SA",
    "BBDC4.SA",
    "ABEV3.SA",
]

BENCHMARKS = [
    "^BVSP",
    "^GSPC",
]

START_DATE = "2015-01-01"

# ---------------------------------------------------------
# SOLUÇÃO: Criar uma sessão customizada para evitar bloqueios
# ---------------------------------------------------------
session = requests.Session()
session.headers.update(
    {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
)

def download_ticker(ticker, retries=3):
    print(f"Baixando {ticker}...")

    for attempt in range(retries):
        try:
            # Passamos a nossa sessão 'disfarçada' para o yfinance
            ativo = yf.Ticker(ticker, session=session)
            df = ativo.history(start=START_DATE, auto_adjust=True)

            if not df.empty:
                # Remove o acento circunflexo de índices como ^BVSP para não gerar nomes de ficheiro inválidos
                nome_ficheiro = ticker.replace('^', '')
                filepath = os.path.join(DATA_DIR, f"{nome_ficheiro}.parquet")

                df.to_parquet(filepath)
                print(f"✓ {ticker} salvo em {filepath}")
                return True
            else:
                print(f"Tentativa {attempt+1}: O dataframe de {ticker} retornou vazio.")

        except Exception as e:
            print(f"Tentativa {attempt+1} falhou para {ticker}: {e}")

        # Aumentar um pouco o tempo de espera ajuda a não ativar as defesas do Yahoo
        time.sleep(3)

    print(f"✗ Falha definitiva ao baixar {ticker}")
    return False

if __name__ == "__main__":
    for ticker in TICKERS:
        download_ticker(ticker)

    for benchmark in BENCHMARKS:
        download_ticker(benchmark)

    print("\nDownload finalizado.")