import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import sys
import os

TARGET_COLUMN_NAME = "Churn"
TEST_DATASET_PATH = "apartments_for_rent_final_test.csv"
MODEL_PATH = "model.joblib"

MOCK_PATH = "data/lab2_2025_dataset.csv"
MOCK_MODEL_PATH = "src/model.joblib"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso correto: python deployed.py <caminho_para_csv>")
        sys.exit(1)

    test_dataset_path = sys.argv[1]

    if not os.path.isfile(test_dataset_path):
        print(f"Erro: Arquivo '{test_dataset_path}' não encontrado.")
        sys.exit(1)

    try:
        loaded_model = joblib.load(MODEL_PATH)
        df = pd.read_csv(test_dataset_path).dropna()

    except FileNotFoundError as e:
        print(
            "Erro: Modelo não encontrado. Certifique-se de que 'model.joblib' está presente."
        )
        sys.exit(1)
    except Exception as e:
        print(f"Erro ao carregar o dataset ou modelo: {e}")
        sys.exit(1)

    X = df.drop(TARGET_COLUMN_NAME, axis=1)
    y = df[TARGET_COLUMN_NAME]

    y_pred = loaded_model.predict(X)

    accuracy = accuracy_score(y, y_pred)
    print(f"Acurácia: {accuracy}")
