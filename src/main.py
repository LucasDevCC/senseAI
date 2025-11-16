# main.py
from treino import treinar_modelo
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# Exemplo de dados (substitua pelos seus)
X = [
    "gostei muito",
    "odiei",
    "achei excelente",
    "péssimo",
    "muito bom",
    "horrível"
]
y = [1, 0, 1, 0, 1, 0]  # 1 = positivo, 0 = negativo

# 1) vetorizar textos -> matrizes numéricas
vectorizer = CountVectorizer()
X_vect = vectorizer.fit_transform(X)  # retorna sparse matrix

# 2) dividir em treino/teste
X_treino, X_test, y_treino, y_test = train_test_split(X_vect, y, test_size=0.33, random_state=42)

# 3) chamar função de treino
treinar_modelo(X_treino, y_treino, X_test, y_test)
