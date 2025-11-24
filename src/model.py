from sklearn.naive_bayes import MultinomialNB

def criar_modelo():
    """
    Cria e retorna o modelo de classificação Naive Bayes.
    """
    modelo = MultinomialNB()
    return modelo
