import argparse
import joblib
import pandas as pd
import sys

from train import merge_data, Log1pColumns

def main():
    # faz o parsing dos argumentos de entrada do script
    parser = argparse.ArgumentParser(description="Script para fazer a predição dos preços")
    parser.add_argument("houses", help="Dataset das casas")
    parser.add_argument("zipcode", help="Dataset de dados demográficos")
    parser.add_argument("output", help="Path para o arquivo a ser criado contendo as predições.")
    args = parser.parse_args()

    # só continua execução se o usuário passou os dois datasets via argumentos da CLI
    if not args.houses or not args.zipcode:
        sys.exit("Você precisa passar dois argumentos: \n 1. Path para o dataset de casas \n 2. Path para o dataset\
                     de dados demográficos")

    # carrega e processa o dataset
    dfh = pd.read_csv(args.houses)
    dfz = pd.read_csv(args.zipcode)
    X = merge_data(dfh, dfz)

    # carrega o pipeline/modelo
    loaded = joblib.load("models/XGBoost_model.joblib")
    model = loaded["model"]
    features = loaded["features"]

    y = model.predict(X[features])
    pd.DataFrame({"price": y}).to_csv(args.output, index=False)

    print(f"O arquivo contendo as predições foi salvo em: {args.output}")

if __name__ == "__main__":
    main()