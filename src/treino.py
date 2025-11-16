from model import criar_modelo
from sklearn.metrics import accuracy_score
import joblib
import os

def treinar_modelo(X_treino, y_treino, X_test, y_test, caminho_modelo='models/model.pkl'):
    # 1) cria o modelo
    modelo = criar_modelo()

    # 2) treina o modelo
    modelo.fit(X_treino, y_treino)

    # 3) avalia o modelo
    preds = modelo.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Acurácia: {acc:.2f}")

    # 4) cria a pasta automaticamente antes de salvar
    os.makedirs(os.path.dirname(caminho_modelo), exist_ok=True)

    # 5) salva o modelo
    joblib.dump(modelo, caminho_modelo)
    print("Modelo salvo em:", caminho_modelo)

    return modelo
