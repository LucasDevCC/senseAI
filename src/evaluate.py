from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import joblib


# 1: GRÁFICO DISTRIBUIÇÃO DOS SENTIMENTOS
def grafico_distribuicao_sentimentos(y_true):
    os.makedirs("reports/figures", exist_ok=True)

    df_temp = pd.DataFrame({"Sentimento": y_true})

    plt.figure(figsize=(6,4))
    sns.countplot(data=df_temp, x="Sentimento", palette="viridis")
    plt.title("Distribuição dos Sentimentos")
    plt.savefig("reports/figures/distribuicao_sentimentos.png")
    plt.close()


# 2: GRÁFICO MÉTRICAS POR CLASSE
def grafico_metricas_por_classe(class_report):
    os.makedirs("reports/figures", exist_ok=True)

    classes = list(class_report.keys())[:2]  # negativo / positivo
    precisao = [class_report[c]["precision"] for c in classes]
    recall = [class_report[c]["recall"] for c in classes]
    f1 = [class_report[c]["f1-score"] for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    plt.figure(figsize=(8,5))
    plt.bar(x - width, precisao, width, label='Precisão')
    plt.bar(x, recall, width, label='Recall')
    plt.bar(x + width, f1, width, label='F1-Score')

    plt.xticks(x, classes)
    plt.ylabel("Valor")
    plt.title("Métricas por Classe")
    plt.legend()
    plt.savefig("reports/figures/metricas_por_classe.png")
    plt.close()


# 3: GRÁFICO TOP 20 PALAVRAS (TF-IDF)
def grafico_top20_palavras(df_texto):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from collections import Counter
    import re
    import nltk

    nltk.download("stopwords")
    from nltk.corpus import stopwords
    stop = set(stopwords.words("portuguese"))

    # LISTA DE PALAVRAS QUE REALMENTE EXPRESSAM SENTIMENTO
    palavras_sentimento = {
        "gostei", "gosto", "amei", "ótimo", "bom", "maravilhoso", "excelente",
        "recomendo", "perfeito", "positivo", "incrível", "top",
        "ruim", "horrível", "péssimo", "lixo", "negativo", "decepcionado",
        "defeito", "terrível", "não", "nunca"
    }

    textos = " ".join(df_texto.astype(str).tolist()).lower()
    textos = re.sub(r"[^a-zA-Zà-úÀ-Ú0-9 ]", " ", textos)
    palavras = textos.split()

    # REMOVE STOPWORDS E PALAVRAS NEUTRAS
    palavras_filtradas = [
        p for p in palavras
        if p not in stop and len(p) > 2 and p in palavras_sentimento
    ]

    contagem = Counter(palavras_filtradas)
    top20 = contagem.most_common(20)

    palavras_plot = [p[0] for p in top20]
    frequencias = [p[1] for p in top20]

    plt.figure(figsize=(10,6))
    sns.barplot(x=frequencias, y=palavras_plot, palette="viridis")
    plt.title("Top 20 Palavras de Opinião (Sentimento)")
    plt.xlabel("Frequência")
    plt.ylabel("Palavra")
    plt.savefig("reports/figures/top20_palavras.png", bbox_inches="tight")
    plt.close()



# 4: GRÁFICO CURVA ROC
def grafico_roc(modelo, X_test, y_test):
    os.makedirs("reports/figures", exist_ok=True)

    plt.figure(figsize=(6,5))
    RocCurveDisplay.from_estimator(modelo, X_test, y_test)
    plt.title("Curva ROC")
    plt.savefig("reports/figures/curva_roc.png")
    plt.close()


# 5: FUNÇÃO PRINCIPAL DE AVALIAÇÃO DO MODELO
def avaliar_modelo(modelo, X_test, y_test, vectorizer, X_test_textos):



    preds = modelo.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")

    print("\n RESULTADOS DO MODELO")
    print("Acurácia:", acc)
    print("F1-Score:", f1)

    class_report_dict = classification_report(y_test, preds, output_dict=True)
    print("\nClassification Report:\n", classification_report(y_test, preds))

    os.makedirs("reports/figures", exist_ok=True)

    # a. Matriz de confusão
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
    plt.title("Matriz de Confusão")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.savefig("reports/figures/matriz_confusao.png")
    plt.close()

    # b. Distribuição dos sentimentos
    grafico_distribuicao_sentimentos(y_test)

    # c. Métricas por classe
    grafico_metricas_por_classe(class_report_dict)

    # d. Top 20 palavras TF-IDF
    df_texto = X_test_textos
    grafico_top20_palavras(df_texto)


    # e. Curva ROC
    grafico_roc(modelo, X_test, y_test)

    print("\n Gráficos salvos em reports/figures/")
