import joblib
import os

def test_vectorizer_loads():
    """Testa se o vectorizer foi salvo e pode ser carregado."""
    assert os.path.exists("models/vectorizer.pkl"), "Vectorizer não encontrado"
    vectorizer = joblib.load("models/vectorizer.pkl")
    assert vectorizer is not None, "Erro ao carregar vectorizer"

def test_vectorizer_transforms_text():
    """Testa se o vectorizer converte texto para vetor."""
    vectorizer = joblib.load("models/vectorizer.pkl")
    texto = ["produto excelente, gostei muito"]
    vetor = vectorizer.transform(texto)

    assert vetor.shape[0] == 1, "Vectorizer não gerou vetor de 1 linha"
    assert vetor.shape[1] > 0, "Vectorizer retornou vetor vazio."
