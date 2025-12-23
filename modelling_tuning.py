import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import json

dagshub.init(
    repo_owner="GaryFaldi",
    repo_name="Membangun_model",
    mlflow=True
)

mlflow.set_tracking_uri("https://dagshub.com/GaryFaldi/Membangun_model.mlflow")
mlflow.set_experiment("Airline Passenger Satisfaction - Advance")

mlflow.sklearn.autolog(
    log_models=True,
    log_datasets=True
)

df = pd.read_csv("Airline Passenger Satisfaction_Cleaned/train_cleaned.csv")

X = df.drop("satisfaction", axis=1)
y = df["satisfaction"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10]
}

rf = RandomForestClassifier(random_state=42)
grid = GridSearchCV(
    rf,
    param_grid,
    cv=3,
    n_jobs=-1
)

with mlflow.start_run(run_name="SKILLED_ADVANCED"):
    grid.fit(X_train, y_train)

    y_pred = grid.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")

    mlflow.log_metric("test_accuracy", acc)
    mlflow.log_metric("test_precision", prec)
    mlflow.log_metric("test_recall", rec)

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax)
    plt.tight_layout()
    plt.savefig("training_confusion_matrix.png")
    plt.close()
    mlflow.log_artifact("training_confusion_matrix.png")

    pred_df = pd.DataFrame({
        "actual": y_test.values,
        "predicted": y_pred
    })
    pred_df.to_csv("predictions.csv", index=False)
    mlflow.log_artifact("predictions.csv")

    with open("best_params.json", "w") as f:
        json.dump(grid.best_params_, f, indent=2)
    mlflow.log_artifact("best_params.json")

    print(f"Run selesai. Accuracy: {acc}")
