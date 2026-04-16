# Solução Case Desafio de Data Science

Este repositório contém a solução do candidato **Thiago Miyazaki** para o **Desafio de Data Science**, como parte do processo
seletivo para vaga de **Cientista de Dados Jr.** na **Made in Web**.

---

## Sobre o repositório

Eis a forma em que o repositório está organizado:
- `artefatos/`: contém os entregáveis esperados descritos no arquivo readme do desafio:
  - `01-analise-entendimento-dos-dados.pdf`: descrição detalhada de como foi feita a análise dos dados;
    - Descrição dos datasets;
    - Observações Iniciais e Hipóteses;
    - Merge dos datasets;
    - Análise Exploratória dos Dados
    - Importância das Features;
  - `02-desenvolvimento-do-modelo.pdf`: descrição em alto nível do processo de desenvolvimento do modelo;
    - Confecção do notebook de exploração de dados e treinamento do baseline;
    - Desenvolvimento dos pipelines de treinamento;
    - Features mais importantes;
    - Escolha do modelo;
  - `03-arquitetura-da-solucao.pdf`: diagrama que representa a arquitetura proposta para o deploy do modelo, com explicação de cada camada;
  - `04-aprendizado-continuo.pdf`: descreve estratégias para fazer com que o modelo aprenda com dados novos;
  - `05-comunicacao-com-stakeholders.pdf`: descreve o conteúdo para uma apresentação para stakeholders;

- `data/`: contém os arquivos de dados utilizados no desafio;
  - `kc_house_data.csv`: com dados que descrevem as características físicas dos imóveis, incluindo o preço (target);
  - `zipcode_demographics.csv`: com dados demográficos e socioeconômicos agregados por código postal (zipcode);
  - `future_unseen_examples.csv`: dataset parecido com `kc_house_data.csv`, mas sem a coluna de preço, utilizado para prever os valores após o treinamento.

- `models/`: contém os arquivos `.joblib` dos modelos treinados durante o desenvolvimento da solução;
  - Linear Regression: `LinearRegression_model.joblib`;
  - Random Forest: `RandomForest_model.joblib`;
  - Gradient Boosting: `GradientBoosting_model.joblib`;
  - XGBoost: `XGBoost_model.joblib`;

- `notebooks/`:
  - `eda-visualization-feature-sel-eng-baseline.ipynb`: análise exploratória dos dados, visualização, feature engineering, treinamento e avaliação do baseline (Regressão Linear);
  - `model-explanation.ipynb`: extração de informações que buscam elucidar o comportamento do modelo, como importância das features e SHAP values, que nos auxiliam a entender como o modelo combina as features para computar as previsões;
- `train.py`: script para reproduzir o treinamento dos modelos;
- `predict.py`: script para gerar as previsões a partir do modelo treinado;
- `requirements.txt`: para instalar os pacotes usando `pip` (recomenda-se Python 3.13.5);
- `environment.yaml`: para criar um conda environment (método recomendado);

---

## Baixando os modelos

Você pode treinar os modelos do zero ou baixá-los por este link (Google Drive):
https://drive.google.com/file/d/1k5_zhkscdkbqsHYJXPzIPMK3mwzf7pQB/view?usp=sharing

Não leva muito tempo para treiná-los, mas talvez seja mais rápido baixar os modelos.
Se resolver baixá-los, por favor, descompacte-os na pasta `models/` de modo que a estrutura da pasta seja:

```
models/
    GradientBoosting_model.joblib
    LinearRegression_model.joblib
    RandomForest_model.joblib
    XGBoost_model.joblib
```

Embora, de fato, o que importa é que o XGBoost_model.joblib esteja no lugar certo, pois é o único modelo referenciado
dentro dos scripts.

---

## Recomendação de leitura

1. Fazer a leitura dos artefatos
2. Fazer a leitura dos notebooks, para entender o processo que deu suporte ao desenvolvimento e interpretação do modelo

---

## Geração do ambiente para a execução dos scripts

### Usando conda (recomendado)

Após rodar os comandos, selecionar o interpretador do Python relacionado ao ambiente do conda `desafio-env`

```
$ conda env create -f environment.yaml
$ conda activate desafio-env
```

### Usando pip

Para preparar o ambiente usando `pip` recomenda-se criar uma nova venv do Python na versão `3.13.5`. Para alterar versões do Python recomenda-se usar `pyenv`.

```
$ pip install -r requirements.txt
```
---

## Executando os scripts de treinamento e predição:

Após a ativação do ambiente (venv/conda) podemos executar os scripts de treinamento/predição.

### Treinamento

O script de treinamento recebe dois argumentos de entrada:

1. Path para o dataset que descrevem as características dos imóveis
2. Path para o dataset com os dados demográficos e socioeconômicos agregados por código postal

```
$ python3 train.py ./data/kc_house_data.csv ./data/zipcode_demographics.csv
```

Os modelos treinados são salvos em `models/`.

### Predição

O script de treinamento recebe três argumentos de entrada:

1. Path para o dataset que descrevem as características dos imóveis
2. Path para o dataset com os dados demográficos e socioeconômicos agregados por código postal
3. Path para o local onde deseja salva o CSV de saída com as predições feitas pelo modelo

```
$ python3 train.py ./data/kc_house_data.csv ./data/zipcode_demographics.csv ./output.csv
```

## Considerações finais

Agradeço pela oportunidade de concorrer à vaga e fico à disposição para tirar qualquer dúvida!
Muito obrigado!
