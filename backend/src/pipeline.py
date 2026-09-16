import kagglehub
from pathlib import Path
import pandas as pd
from kagglehub import KaggleDatasetAdapter

RAW_DATA_DIR = Path(__file__).resolve().parents[1]/ "data"/ "raw"
RAW_DATA_DIR.mkdir(parents = True, exist_ok = True)

RAW_FILE_PATH = RAW_DATA_DIR/ "telco_customer_churn.csv"

df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "blastchar/telco-customer-churn",
    "WA_Fn-UseC_-Telco-Customer-Churn.csv",
)

if not RAW_FILE_PATH.is_file():
    df.to_csv(RAW_FILE_PATH, index = False)
    print("File saved to:", RAW_DATA_DIR)

print(df.head())

