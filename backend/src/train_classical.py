from pipeline import NUM_COLS, CAT_COLS, build_preprocessor, clean_data, save_csv, split_data
from kagglehub import dataset_load, KaggleDatasetAdapter, kagglehub
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


if __name__ == "__main__":
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "blastchar/telco-customer-churn",
        "WA_Fn-UseC_-Telco-Customer-Churn.csv",
    )

    # Download dataset if not already downloaded
    save_csv(df, "raw")

    # Clean the dataset
    df = clean_data(df)

    # Save the cleaned data
    save_csv(df, "processed")

    # Split data into X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = split_data(df)

    # Get the pipeline obj
    preprocessor = build_preprocessor(NUM_COLS, CAT_COLS)

    # Pass split data into pipeline
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n--- Classification Report ---\n")
    print(classification_report(y_test, y_pred))
