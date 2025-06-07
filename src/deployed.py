import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import sys

TARGET_COLUMN_NAME = "Churn"
TEST_DATASET_PATH = "apartments_for_rent_final_test.csv"

MOCK_PATH = "data/lab2_2025_dataset.csv"

if __name__ == "__main__":
    try:
        loaded_model = joblib.load("src/model.joblib")
        df = pd.read_csv(TEST_DATASET_PATH).dropna()

    except FileNotFoundError as e:
        print(
            "Arquivo não encontrado. Verifique se o dataset ou o arquivo do modelo estão na mesma pasta deste arquivo"
        )
        sys.exit(1)
    except Exception as e:
        print(f"Erro: {e}")
        sys.exit(1)

    X = df.drop(TARGET_COLUMN_NAME, axis=1)
    y = df[TARGET_COLUMN_NAME]

    y_pred = loaded_model.predict(X)

    accuracy = accuracy_score(y, y_pred)
    print(f"Acurácia: {accuracy}")
