import pandas as pd
import os

settings = {
    "cleaning": {
        "drop_columns": [],       # para se for necessario remover colunas
        "fill_na_method": "ffill",
        "text_normalization": True,
        "convert_dates": False
    }
}

RAW_PATH = "data/raw/dados_vendas_amazon_ml.csv"
PROCESSED_PATH = "data/processed/dados_limpos.csv"

def carregar_dados():
    print("Lendo dados brutos...")
    return pd.read_csv(RAW_PATH)

def limpar_dados(df):
    cfg = settings["cleaning"]

    # Remover colunas
    df = df.drop(columns=cfg["drop_columns"], errors="ignore")

    # Tratar valor nulo
    if cfg["fill_na_method"] == "ffill":
        df = df.fillna(method="ffill").fillna(method="bfill")

    # normalizar textos
    if cfg["text_normalization"]:
        for col in df.select_dtypes(include="object"):
            df[col] = df[col].astype(str).str.strip().str.lower()

    # converter datas
    if cfg["convert_dates"]:
        for col in df.columns:
            if "data" in col.lower():
                df[col] = pd.to_datetime(df[col], errors="coerce")

    return df

def salvar_dados(df):
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Arquivo limpo salvo em: {PROCESSED_PATH}")

def main():
    df_raw = carregar_dados()
    df_limpo = limpar_dados(df_raw)
    salvar_dados(df_limpo)

if __name__ == "__main__":
    main()
