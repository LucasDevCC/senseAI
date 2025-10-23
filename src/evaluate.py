# / Avaliação e Gráficos \
# Integrante: Lucas Lima
# Este script faz uma análise basica do conjunto de dados

# Bibliotecas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# caminho csv para puxar os dados
data_path = os.path.join("data", "raw", "dados_vendas_amazon_ml.csv")

# verifica se existe o arquivo para evitar erros
if not os.path.exists(data_path):
    print("O arquivo não foi encontrado. Verifique se o nome e o caminho estão corretos.")
else:
    # Lendo o arquivo csv
    df = pd.read_csv(data_path, encoding="utf-8")

    # mostrando as primeiras linhas do csv
    print("✅ Arquivo carregado com sucesso!\n")
    print("Primeiras linhas do conjunto de dados:")
    print(df.head(), "\n")

    # mostra informações gerais sobre o dataset
    print("Informações gerais sobre o dataset:")
    print(df.info(), "\n")

    # mostra estatísticas básicas
    print("Resumo estatístico das colunas numéricas:")
    print(df.describe(), "\n")

    # mostra as colunas existentes
    print("Colunas disponíveis no dataset:")
    print(list(df.columns), "\n")

    # escolhe automaticamente uma coluna numérica para o gráfico
    coluna_numerica = None
    for coluna in df.columns:
        if df[coluna].dtype in ["int64", "float64"]:
            coluna_numerica = coluna
            break

    # Criando grafico simples
    if coluna_numerica:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[coluna_numerica], kde=True, color="steelblue")
        plt.title(f"Distribuição da coluna: {coluna_numerica}")
        plt.xlabel(coluna_numerica)
        plt.ylabel("Frequência")
        plt.tight_layout()
        plt.show()
    else:
        print("Nenhuma coluna numérica encontrada para gerar grafico.")
