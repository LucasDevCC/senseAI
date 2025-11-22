import joblib
import os
import numpy as np

def test_model_loads():
    """verifica se o modelo foi salvo corretamente."""
    assert os.path.exists("models/model.pkl"), "modelo não encontrado"
    modelo = joblib.load("models/model.pkl")
    assert modelo is not None, "erro ao carregar o modelo"

def test_model_predicts():
    """Testa se o modelo consegue prever sem erro"""
    modelo = joblib.load("models/model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")

    texto = ["produto muito ruim, não gostei"]
    x = vectorizer.transform(texto)
    pred = modelo.predict(x)

    assert pred is not None, "a previsão falhou."
    assert pred[0] in [0, 1], "o modelo está retornando valor inválido."
