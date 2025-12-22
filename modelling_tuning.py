import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Hubungkan kode ke DagsHub
dagshub.init(repo_owner='GaryFaldi', repo_name='Membangun_model', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/GaryFaldi/Membangun_model.mlflow")
mlflow.set_experiment("Airline Passenger Satisfaction - Advance")

# Load data
df = pd.read_csv("Airline Passenger Satisfaction_Cleaned/train_cleaned.csv")
X = df.drop("satisfaction", axis=1)
y = df["satisfaction"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter Tuning
param_grid = {"n_estimators": [100, 200], "max_depth": [None, 10]}
model = RandomForestClassifier(random_state=42)
grid = GridSearchCV(model, param_grid, cv=3)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

with mlflow.start_run():
    # 1. Log Parameter & Metric (Sesuai Skilled/Advance)
    mlflow.log_params(grid.best_params_)
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", acc)

    # 2. ARTEFAK TAMBAHAN 1: Confusion Matrix (Bentuk Gambar)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax)
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png") # Mengirim gambar ke DagsHub

    # 3. ARTEFAK TAMBAHAN 2: File CSV Prediksi
    pred_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    pred_df.to_csv("predictions.csv", index=False)
    mlflow.log_artifact("predictions.csv") # Mengirim file CSV ke DagsHub

    # 4. Log Model Folder
    mlflow.sklearn.log_model(sk_model=best_model, artifact_path="model")

    print(f"Accuracy: {acc}")