import argparse
import joblib
import numpy as np
import pandas as pd
import sys
import warnings

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from typing import Any
from xgboost import XGBRegressor

warnings.filterwarnings(
    "ignore",
    message="`sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel`"
)

LREG = "LinearRegression"
RFOREST = "RandomForest"
GBOOST = "GradientBoosting"
XGBOOST = "XGBoost"
RANDOM_STATE = 42

# features com variáveis binárias - separamos porque não devem passar pelo StandardScaler
BINARY_COLS = ["waterfront", "was_renovated"]

# estas features possuem uma distribuição assimétrica e vamos usar log-transform nelas
SKEWED_FEATURE_COLS = [
    "sqft_living",
    "sqft_basement",
    "sqft_lot",
    "sqft_above",
    "sqft_living15",
    "sqft_lot15",
    "ppltn_qty",
    "medn_hshld_incm_amt",
    "medn_incm_per_prsn_amt",
]

# features a serem dropadas do dataset de zipcode - já temos colunas percentuais sobre elas
ZIPCODE_DROP_COLS = [
    "urbn_ppltn_qty",
    "sbrbn_ppltn_qty",
    "farm_ppltn_qty",
    "non_farm_qty",
    "edctn_less_than_9_qty",
    "edctn_9_12_qty",
    "edctn_high_schl_qty",
    "edctn_some_clg_qty",
    "edctn_assoc_dgre_qty",
    "edctn_bchlr_dgre_qty",
    "edctn_prfsnl_qty",
    "hous_val_amt",
]

# parâmetros de busca para a otimização de hiperparâmetros dos modelos baseados em árvore
# - estimativas: [conservadores, médias, agressivas]
SEARCH_PARAMS = {
    LREG: {
    },
    RFOREST: {
        "model__regressor__n_estimators": [200, 400, 600],       # qtd. de árvores
        "model__regressor__max_depth": [None, 10, 20, 30],       # profundidade das árvores (none = ilimitado)
        "model__regressor__min_samples_split": [2, 5, 10],       # qtd. mín. de samples em cada nó para permitir split
        "model__regressor__min_samples_leaf": [1, 2, 4],         # qtd. de samples que uma folha pode ter (dps do split)
        "model__regressor__max_features": ["sqrt", "log2", 0.5], # política de escolha dos atributos a cada split
    },                                                           # - raiz do total, log2 do total, ou metade
    GBOOST: {
        "model__regressor__n_estimators": [100, 200, 300],          # qtd. de árvs - tem que ser menor pq é sequencial
        "model__regressor__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__regressor__max_depth": [2, 3, 4],
        "model__regressor__min_samples_split": [2, 5, 10],
        "model__regressor__min_samples_leaf": [1, 2, 4],
        "model__regressor__subsample": [0.6, 0.8, 1.0],             # % dos dados que cada árvore deve enxergar
    },
    XGBOOST: {
        "model__regressor__n_estimators": [200, 400, 600],
        "model__regressor__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__regressor__max_depth": [3, 4, 5, 6],
        "model__regressor__min_child_weight": [1, 3, 5],        # parecido com min_samples_leaf
        "model__regressor__subsample": [0.6, 0.8, 1.0],         # % dos dados (linhas) que cada árvore deve enxergar
        "model__regressor__colsample_bytree": [0.6, 0.8, 1.0],  # % dos dados (coluna) que cada árvore deve enxergar
        "model__regressor__gamma": [0, 0.1, 0.3],               # qual deve ser o ganho para permitir split
        "model__regressor__reg_alpha": [0, 0.01, 0.1, 1],       # alpha dos coeficientes (regularização lasso)
        "model__regressor__reg_lambda": [1, 2, 5, 10],          # lambda dos coeficientes (regularização ridge)
    }
}

class Log1pColumns(BaseEstimator, TransformerMixin):
    """
    Aplica log1p às colunas selecionadas.
    Herda set_params() e get_params() de BaseEstimator.
    Herda fit_transform() de TransformerMixin

    Atributos:
    - columns: lista de colunas a serem transformadas (passada na inicialização)
    - columns_: colunas definidas no fit após leitura do que foi passado no construtor
    """

    def __init__(self, columns):    # convenção
        self.columns = columns      # atributo pré-fit
        self.columns_ = None        # atributo pós-fit

    def fit(self, X, y=None):
        # adiciona o nome das colunas em self.columns_
        self.columns_ = [col for col in self.columns if col in X.columns]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # aplica log-transform com clip (evitando valor = 0)
        X = X.copy()
        for col in self.columns_:
            X[col] = np.log1p(X[col].clip(lower=0))
        return X

def make_linear_pipeline(ds_columns):
    cont_cols = [col for col in ds_columns if col not in BINARY_COLS]
    scaler = ColumnTransformer(
        transformers=[
            ("scale_cont", StandardScaler(), cont_cols),
            ("keep_bin", "passthrough", BINARY_COLS)
        ],
        verbose_feature_names_out=False
    ).set_output(transform="pandas")

    # threshold escolhido pela análise dos coeficientes no notebook
    selector = SelectFromModel(
        estimator=LassoCV(cv=5, random_state=RANDOM_STATE, max_iter=20000),
        threshold=0.003
    )

    # TransformedTargetRegressor aplica transformação no target na entrada do modelo e restora na saída
    regressor = TransformedTargetRegressor(
        regressor=LinearRegression(),
        func=np.log1p,
        inverse_func=np.expm1,
    )

    return Pipeline(
        steps=[
            ("log_transform", Log1pColumns(SKEWED_FEATURE_COLS)),
            ("scaler", scaler),
            ("selector", selector),
            ("model", regressor)
        ]
    )

def make_tree_pipeline(regressor: Any) -> Pipeline:
    model_regressor = TransformedTargetRegressor(
        regressor=regressor,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )

    return Pipeline(
        steps=[
            ("log_features", Log1pColumns(SKEWED_FEATURE_COLS)),
            ("model", model_regressor),
        ]
    )

def merge_data(dfh, dfz):
    # dropa linhas com dados faltantes e dropa a coluna id
    dfh = dfh.dropna().copy()
    dfh = dfh.drop(columns=["id"])

    # constrói as features year_month e house_age
    dfh["date"] = pd.to_datetime(dfh["date"], format="%Y%m%dT%H%M%S")
    dfh["year_month"] = dfh["date"].dt.year * 100 + dfh["date"].dt.month
    dfh["house_age"] = dfh["date"].dt.year - dfh["yr_built"]
    dfh = dfh.drop(columns=["date", "yr_built"])

    # constrói a feature binária was_renovated
    dfh["was_renovated"] = (dfh["yr_renovated"] > 0).astype(int)
    dfh = dfh.drop(columns=["yr_renovated"])

    # dropa variáveis numéricas absolutas em favor dos percentuais
    # e dropa hous_val_amt por risco de multicolinearidade/data leakage
    dfz = dfz.drop(columns=ZIPCODE_DROP_COLS)

    # faz o merge dos dfs
    merged = dfh.merge(dfz, on="zipcode", how="left")
    merged = merged.drop(columns=["zipcode"])

    return merged

def train_and_evaluate_models(X_train, X_test, y_train, y_test, pipelines):
    """
    Recebe os pipelines dos modelos, faz a otimização de parâmetros, treina os modelos,
    e retorna as métricas dos melhores modelos com seus pipelines.

    :param X_train: dataset de treino
    :param X_test: dataset de teste
    :param y_train: sinal supervisionado do dataset de treino
    :param y_test: sinal supervisionado do dataset de teste
    :param pipelines: os pipelines a serem utilizados durante o treino / otimização de hiperparâmetros
    :return results: dicionário com os resultados dos modelos:
            "best_params": melhores hiperparâmetros dos modelos
            "cv_r2": melhor R² obtido no cross-validation durante a otimização de hiperparâmetros,
            "test_r2": R² score do modelo,
            "test_mae": MAE (mean absolute error) do modelo,
            "test_rmse": RMSE (root mean squared error) do modelo,
    :return fitted_models: dicionário com os pipelines de cada modelo
    """
    results = {}
    fitted_models = {}

    for model_name, pipeline in pipelines.items():
        print(f"\n===== Treinando: {model_name} =====")

        # pega os parâmetros a serem utilizado pela otimização de hiperparâmetros
        params = SEARCH_PARAMS[model_name]

        # não há otimização para o LinearRegression
        if not params:
            pipeline.fit(X_train, y_train)
            best_estimator = pipeline
            best_params = {}
            best_cv_score = None

        else:
            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=params,     # dicionário com parâmetros e seus valores possíveis
                n_iter=30,                      # número de combinações de parâmetros a testar
                scoring="r2",
                cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
                random_state=RANDOM_STATE,
                n_jobs=-1                       # usa todos os cores disponíveis
            )

            # inicia busca pelos melhores hiperparâmetros
            search.fit(X_train, y_train)

            best_estimator = search.best_estimator_     # melhor pipeline encontrado durante a otimização
            best_params = search.best_params_           # melhores parâmetros encontrados
            best_cv_score = search.best_score_          # melhor score R² encontrado durante cross-validation

        y_pred = best_estimator.predict(X_test)

        results[model_name] = {
            "best_params": best_params,
            "cv_r2": best_cv_score,
            "test_r2": r2_score(y_test, y_pred),
            "test_mae": mean_absolute_error(y_test, y_pred),
            "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        }

        fitted_models[model_name] = best_estimator

        print("Best params:", best_params)
        print("CV R²:", best_cv_score)
        print("Test R²:", results[model_name]["test_r2"])
        print("Test MAE:", results[model_name]["test_mae"])
        print("Test RMSE:", results[model_name]["test_rmse"])

    return results, fitted_models

def main():
    """
    Diferentemente do que foi feito para o treino do baseline,
    modelos baseados em árvore requerem menos feature engineering,
    então vamos criar um dataframe base e daí derivar tipos diferentes
    de processamento.
    """

    # faz o parsing dos argumentos de entrada do script
    parser = argparse.ArgumentParser(description="Script para executar o treinamento dos modelos e \
        salvar pipeline + métricas")
    parser.add_argument("houses", help="Dataset das casas")
    parser.add_argument("zipcode", help="Dataset de dados demográficos")
    args = parser.parse_args()

    # só continua execução se o usuário passou os dois datasets via argumentos da CLI
    if not args.houses or not args.zipcode:
        sys.exit("Você precisa passar dois argumentos: \n 1. Path para o dataset de casas \n 2. Path para o dataset\
                 de dados demográficos")

    # carrega os datasets, dropa price, e cria a série target com os valores de price
    dfh = pd.read_csv(args.houses)
    dfz = pd.read_csv(args.zipcode)
    df = merge_data(dfh, dfz)
    X = df.drop(columns=['price'])
    y = df['price']

    # split dos dados
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    # cria os pipelines lineares e baseados em árvore
    models = {
        LREG: make_linear_pipeline(list(X.columns)),
        RFOREST: make_tree_pipeline(RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
        GBOOST: make_tree_pipeline(GradientBoostingRegressor(random_state=RANDOM_STATE)),
        XGBOOST: make_tree_pipeline(
            XGBRegressor(
                random_state=RANDOM_STATE,
                n_jobs=-1,
                objective="reg:squarederror"
            )
        )
    }

    # treina os modelos e pega os resultados
    results, best_models = train_and_evaluate_models(X_train, X_test, y_train, y_test, models)

    # salva os modelos em /models
    for name, pipeline in best_models.items():
        print(f"Salvando modelo em models/{name}_model.joblib")
        joblib.dump({
            "model": pipeline,
            "features": X.columns.tolist(),
            "best_params": results[name]["best_params"],
            "metrics":{
                "cv_r2": results[name]["cv_r2"],
                "test_r2": results[name]["test_r2"],
                "test_mae": results[name]["test_mae"],
                "test_rmse": results[name]["test_rmse"],
            }
        }, f"models/{name}_model.joblib")

if __name__ == "__main__":
    main()
