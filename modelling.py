import mlflow
import mlflow.sklearn
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import os

# Initialize DagsHub
dagshub.init(repo_owner='ivan8419', repo_name='SMSML_Muhammad_Ivan', mlflow=True)

mlflow.set_experiment("Credit_Card_Fraud_Detection")

# Disable autolog to have more control as requested by reviewer for specific paths
# mlflow.sklearn.autolog()

def load_data():
    dataset_path = "creditcard_preprocessed.csv"
    if not os.path.exists(dataset_path):
        # Try parent directory if not found (for consistency with notebook paths)
        dataset_path = "../namadataset_preprocessing/creditcard_preprocessed.csv"
    
    if not os.path.exists(dataset_path):
         # If still not found, check if it's in the same folder as the script
         dataset_path = os.path.join(os.path.dirname(__file__), "creditcard_preprocessed.csv")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found. Please ensure creditcard_preprocessed.csv is available.")
        
    df = pd.read_csv(dataset_path)
    X = df.drop("Class", axis=1)
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test

def train_model():
    X_train, X_test, y_train, y_test = load_data()

    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)

        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # Explicitly log the model as requested by Criterion 2
        mlflow.sklearn.log_model(model, "best_random_forest_model")

        print("Training completed and logged to DagsHub.")
        print(f"Metrics - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

if __name__ == "__main__":
    train_model()