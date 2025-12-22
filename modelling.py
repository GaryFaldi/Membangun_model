import pandas as pd
import mlflow
import mlflow.sklearn
import os
import dagshub
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- PERBAIKAN UNTUK CI & DAGSHUB ---
# Gunakan environment variables agar kredensial aman di GitHub Secrets
dagshub_url = "https://dagshub.com/GaryFaldi/Membangun_model.mlflow"

# Inisialisasi DagsHub agar GitHub Actions bisa login otomatis
dagshub.init(repo_owner='GaryFaldi', repo_name='Membangun_model', mlflow=True)

mlflow.set_tracking_uri(dagshub_url)
mlflow.set_experiment("Airline Passenger Satisfaction - CI")

# Sesuaikan path dataset karena nanti di folder MLProject
# Pastikan file CSV ada di dalam folder MLProject
df = pd.read_csv("Airline Passenger Satisfaction_Cleaned/train_cleaned.csv") 

X = df.drop("satisfaction", axis=1)
y = df["satisfaction"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.autolog()

with mlflow.start_run():
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Accuracy:", acc)