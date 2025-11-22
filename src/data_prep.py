import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

RAW_PATH = "data/raw/dados_vendas_amazon_ml.csv"
PROCESSED_PATH = "data/processed/dados_limpos.csv"


def carregar_e_limpar_dados():
    """
    Carrega o CSV bruto e faz limpeza simples:
     deixa texto minusculo
     remove valores nulos
     renomeia colunas
    """
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"Arquivo {RAW_PATH} não encontrado!")

    df = pd.read_csv(RAW_PATH)

    # Identifica coluna de texto
    col_texto = [col for col in df.columns if df[col].dtype == "object"][0]
    df[col_texto] = df[col_texto].astype(str).str.lower().str.strip()

    # Remove nulos
    df = df.dropna()

    # Salva arquivo limpo
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"✔ Arquivo limpo salvo em: {PROCESSED_PATH}")

    return df, col_texto


def preparar_dados_para_treino():

    df = pd.read_csv("data/processed/dados_limpos.csv")

    col_texto = "Review"
    col_y = "Sentiment"

    X_texto = df[col_texto]           # textos originais
    y = df[col_y]

    vectorizer = TfidfVectorizer(max_features=5000)
    X_tfidf = vectorizer.fit_transform(X_texto)

    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42
    )

    # também retorno os textos originais separados!
    X_train_texto, X_test_texto = train_test_split(
        X_texto, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, vectorizer, X_test_texto