# senseAI

Projeto SenseAI

Nesse projeto realizamos o desenvolvimento de uma Inteligência artificial capaz de analisar os Reviews de clientes dos produtos da Amazon, classificando cada mensagem como positiva ou negativa..

O modelo foi treinado utilizando um conjunto de dados com milhares de avaliações de produtos da Amazon com o objetivo de permitir que um sistema automatizado consiga entender a opnião dos usuários de forma rápida e escalável.

Nesse foco construímos uma pipeline completa que inclui os seguintes passos:

\ Limpeza e pré-processamento dos textos \
\ Vetorização com TF-IDF \
\ Treinamento de um classificador Naive Bayes \
\ Avaliação do modelo com métricas e gráficos \ 
\ Geração de modelo e vetorizador para reutilização \

Além do modelo também foi gerado alguns gráficos para análise dos resultados, permitindo visualizar o desempenho da IA e padrões dentro dos dados.

Nosso objetivo:

Foi criar um sistema de classificação de sentimentos capaz de identificar automaticamente se uma avaliação escrita
representa uma opnião positiva ou negativa, auxiliando empresas na análise de feedbacks em larga escala.

Etapas do modelo:

1. Coleta dos dados

 Utilizamos um Dataset com avaliações textuais de produtos (data\processed\dados_limpos.csv ou dados_vendas_amazon.csv)

2. Pré-processamento

Normalização de texto
Remoção de stopwords
Conversão para minusculas
Limpeza de caracteres especiais

3. Extração das caracteristicas

Cada review é convertido em uma matriz numérica que representa a importancia de cada palavra no texto

4. Treinamento do modelo

Foi utilizado o classificador Multinomial Naive Bayes, amplamente empregado para as tarefas de PLN.

5. Avaliação:
 Metricas que foram calculadas:

\ Acurácia \
\  F1-Score \
\ Report completo por classe \
\ Matriz de confusão \

6. Visualização dos resultados:
  Foi gerado cinco graficos que auxiliam na interpretação do modelo:
   Distribuição de sentimentos
   Matriz de confusão
   Curva ROC
   Métricas por classe
   Top 20 palavras mais frequentes na avaliação

Resultados finais:

O modelo apresentou o seguinte resultado:

Acurácia final: 0.99
F1-Score: 0.99

Esses valores mostram que o modelo é altamente eficaz em identificar corretamente sentimentos positivos e negativos.

Os graficos estão presentes em reports\figures.

# Conclusão

O projeto mostra que técnicas de Processamento de Linguagem Natural combinadas com modelos estatísticos pode ser extremamente eficientes na classificação de comentários.

A IA construída é capaz de:

 Ler automaticamente uma avaliação  
 Extrair o sentimento do texto  
 Ajudar empresas a entender a opinião dos clientes  
 Automatizar análises que antes eram manuais  

Além disso, os gráficos gerados permitem uma compreensão visual clara da performance do modelo e dos padrões existentes nas avaliações.



Como rodar o projeto:

Após ter as bibliotecas instaladas digitar no terminal:

pip install -r requirements.txt
python -m src.main

---------------------------------------------------------------------------------------------------------------------------------------

Integrante 2 (Lucas Lima RA 2224100547):
Avaliações de dados utilizando métricas e gráficos;

O que eu fiz
Script de avaliação ('src/evaluate.py') que:
 lê 'data/processed/dados_limpos.csv';
 calcula accuracy, precision, recall, f1;
 salva métricas em 'reports/figures';
 salva matriz de confusão em 'reports/figures/'.
Preenchimento dos notebooks que:
 Realiza a análise exploratória de dados;
 Mostra alguns gráficos com números baseado no arquivo CSV já limpo
Preenchimento do main.py que:
 Realiza a execução do código e faz os calculos e gráficos;
 Mostra as métricas automatizadas;
 Faz o pipeline funcionar corretamente;

-------------------------------------------------------------------------------------------------------------------------------------
Integrante 4 (Guilherme Henrique 2225107272)
Organização e entrega final

Oque eu fiz 
- Revisei códigos 
- Montei o roteiro para apresentação
- Atualizei o README.md
- Mantive as pastas docs ARCHITECTURE.md e ROADMAP.md atualizadas

-------------------------------------------------------------------------------------------------------------------------------------

Integrante 3 (Matheus Dourado Batista 2224101507)

O que eu fiz

Implementei o modelo em src/model.py

Realizei o treino em src/train.py

Ajustei hiperparâmetros e validei as métricas

Mantive os arquivos src/model.py, src/train.py e models/model.pkl organizados e atualizados


-------------------------------------------------------------------------------------------------------------------------------------

Integrante 1 (Lucas Dias Rocha 2224102285)

Coleta, Limpeza e Preparação dos Dados 

  O que eu fiz

- Organização da estrutura de dados (data/):
- Coloquei os arquivos brutos em data/raw/, incluindo o arquivo principal dados_vendas_amazon_ml.csv.
- Preparei o diretório data/processed/, onde será salvo o arquivo já limpo e padronizado (dados_limpos.csv).
- Desenvolvimento do script de preparação, que:
- Lê os dados originais dentro de data/raw/;
- Remove colunas desnecessárias;
- Corrige tipos de dados incorretos (datas, números, textos);
- Normaliza textos (case, remoção de caracteres especiais, etc.);
- Gera e salva o arquivo limpo data/processed/dados_limpos.csv.
- Mostra estatísticas iniciais do dataset (linhas, colunas, tipos);
- Exibe quantidade de valores nulos e duplicados;
- Apresenta gráficos simples para entender a distribuição dos dados;

 
