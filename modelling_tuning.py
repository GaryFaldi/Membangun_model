import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment("Airline Passenger Satisfaction - Skilled")

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

model = RandomForestClassifier(random_state=42)
grid = GridSearchCV(model, param_grid, cv=3)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

with mlflow.start_run():
    mlflow.log_params(grid.best_params_)
    
    mlflow.log_metric("best_cv_score", grid.best_score_)

    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy", acc)
    
    input_example = X_train.head(1)
    mlflow.sklearn.log_model(
        sk_model=best_model, 
        artifact_path="model",
        input_example=input_example
    )

    print(f"Model Tuning Berhasil! Accuracy: {acc}")