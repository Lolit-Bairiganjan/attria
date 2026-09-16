import kagglehub
from pathlib import Path
import pandas as pd
from kagglehub import KaggleDatasetAdapter
from sklearn.model_selection import train_test_split

def save_csv(df: pd.DataFrame, stage: str = "processed"):
    if stage == "raw":
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
        RAW_DATA_DIR.mkdir(parents = True, exist_ok = True)

        df.to_csv(PROCESSED_FILE_PATH, index=False)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:

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

    # Map simple yes/no in Churn to 1/0 respectively
    if "Churn" in df.columns and df["Churn"].dtype == "object":
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    print(f"\n{df.shape[0]} rows loaded\n")

    # Print clean data
    print("\n============\nCleaned Data\n============")
    print(df.head())

    return df

def split_data(
        df: pd.DataFrame,
        target_col: str = "Churn",
        test_size: float = 0.2,
        random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    # Clean data before splitting
    df = clean_data(df)

    y = df[target_col]
    X = df.drop(columns=[target_col])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test


# Numerical features (to be scaled)
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

# Categorical features (to be One-Hot Encoded)
CAT_COLS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

if __name__ == "__main__":
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "blastchar/telco-customer-churn",
        "WA_Fn-UseC_-Telco-Customer-Churn.csv",
    )

    save_csv(df, "raw")
    X_train, X_test, y_train, y_test = split_data(df)
