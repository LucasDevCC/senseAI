O architecture é o documento que descreve a arquitetura geral do projeto de classificação de sentimentos com IA.

 📁 Estrutura de Diretórios

senseAI/
│
├── data/
│ ├── raw/ # Dados brutos
│ └── processed/ # Dados limpos usados pelo modelo
│
├── docs/ # Documentação do projeto
│ ├── ARCHITECTURE.md
│ └── ROADMAP.md
│
├── models/ # Artefatos do modelo treinado
│ ├── model.pkl
│ └── vectorizer.pkl
│
├── notebooks/ # Notebooks de análise e experimentos
│
├── reports/
│ └── figures/ # Gráficos gerados automaticamente
│
├── src/
│ ├── data_prep.py # Pré-processamento e divisão de dados
│ ├── treino.py # Treinamento do modelo
│ ├── evaluate.py # Avaliação e gráficos
│ ├── model.py # Criação do modelo
│ └── main.py # Pipeline principal
│
└── tests/ # Testes automatizados


# Fluxo do Pipeline

1. Carregamento dos dados (data/raw)
2. Processamento e limpeza (data_prep.py)
3. **Vetorização TF-IDF**
4. Treinamento do modelo (treino.py)
5. Avaliação do modelo (evaluate.py)* 
    matriz de confusão  
    distribuição dos sentimentos  
    métricas por classe  
    curva ROC  
    top 20 palavras  
6. Geração de gráficos
7. Salvamento dos artefatos em models/
8. Testes automatizados

---

# Componentes Principais

# data_prep.py
Limpa os textos, remove ruídos e divide os dados.

# treino.py
Treina o algoritmo Multinomial Naive Bayes e salva 'model.pkl`.

# model.py
Função que cria o modelo.

# evaluate.py
Calcula métricas e gera 5 gráficos.

# main.py
Executa todo o pipeline com o comando:
python -m src.main
(obs: não esquecer de instalar as bibliotecas com python -m requirements.txt para evitar erros)

# tests/
Testes simples para garantir funcionamento do modelo.

---

# Tecnologias Usadas
 Python  
 Pandas  
 Scikit-learn  
 NLTK  
 Matplotlib  
 Seaborn  
 Pytest  

# Conclusão
A arquitetura segue boas práticas de IA, garantindo organização, escalabilidade e fácil manutenção.
