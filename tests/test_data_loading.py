import pandas as pd
import os

def test_csv_exists():
    """verifica se o arquivo CSV já processado existe"""
    assert os.path.exists("data/processed/dados_limpos.csv"), "Arquivo dados_limpos.csv não encontrado"

def test_csv_has_required_columns():
    """Confere se o CSV possui as colunas esperadas"""
    df = pd.read_csv("data/processed/dados_limpos.csv")
    required_cols = ["Review", "Sentiment"]

    for col in required_cols:
        assert col in df.columns, f"Coluna obrigatória ausente: {col}"
