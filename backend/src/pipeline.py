import kagglehub
from pathlib import Path
import pandas as pd
from kagglehub import KaggleDatasetAdapter

def save_csv(df: pd.DataFrame, where: int):
    if where == 0:
        RAW_DATA_DIR = Path(__file__).resolve().parents[1]/ "data"/ "raw"
        RAW_DATA_DIR.mkdir(parents = True, exist_ok = True)
        
        RAW_FILE_PATH = RAW_DATA_DIR/ "telco_customer_churn.csv"
        
        if not RAW_FILE_PATH.is_file():
            df.to_csv(RAW_FILE_PATH, index = False)
            print("File saved to:", RAW_DATA_DIR)

    else:
        PROCESSED_FILE_PATH = (
        Path(__file__).resolve().parents[1]
            / "data"
            / "processed"
            / "cleaned_telco.csv"
        )
        df.to_csv(PROCESSED_FILE_PATH, index=False)


def clean_data(df: pd.DataFrame):

    # Remove unnecesarry col_name -> CustomerId
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Convert TotalCharges into float64
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = df["TotalCharges"].apply(pd.to_numeric, errors = "coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(0)

        if df["TotalCharges"].dtype == "float64":
            print("\nSuccessfully converted to float64 datatype\n")
        else:
            print("Error")

    print(df.info())
    print(df.isnull().sum())

    print(f"\n{df.shape[0]} rows loaded\n")

    # Print clean data
    print("\n============\nCleaned Data\n============")
    print(df.head())
    # print(df.info())
    # print(df.shape)
    # print(df.dtypes)
    # print(df.isnull().sum())
    return df

if __name__ == "__main__":
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "blastchar/telco-customer-churn",
        "WA_Fn-UseC_-Telco-Customer-Churn.csv",
    )

    save_csv(df, 0)
    df = clean_data(df)
    save_csv(df, 1)
