from src.data_prep import preparar_dados_para_treino
from src.treino import treinar_modelo
from src.evaluate import avaliar_modelo
import joblib
import os

def main():

    print("\n INICIANDO PIPELINE COMPLETO \n")

    # 1) Preparar dados
    X_train, X_test, y_train, y_test, vectorizer, X_test_textos = preparar_dados_para_treino()

    # 2) Treinar modelo
    modelo = treinar_modelo(X_train, y_train, X_test, y_test)

    # 3) Avaliar modelo (agora corretamente)
    avaliar_modelo(modelo, X_test, y_test, vectorizer, X_test_textos)

    # 4) Salvar artefatos
    os.makedirs("models", exist_ok=True)
    joblib.dump(modelo, "models/model.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")

    print("\n Pipeline concluído com sucesso.")

if __name__ == "__main__":
    main()
