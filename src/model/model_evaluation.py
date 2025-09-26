import os
import json
import yaml
import pickle
import logging
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple

# -------------------- logging --------------------
logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
file_handler = logging.FileHandler("model_evaluation_errors.log")
file_handler.setLevel(logging.ERROR)
fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(fmt)
file_handler.setFormatter(fmt)
if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# -------------------- config --------------------
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://ec2-54-159-96-0.compute-1.amazonaws.com:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "dvc-pipeline-runs-ue1")  # NEW name to avoid legacy artifact root
ARTIFACT_ROOT = os.getenv("MLFLOW_ARTIFACT_ROOT", "s3://mlflow-bucket-102")  # your existing bucket in us-east-1

# Optional but useful if running from your laptop
# Ensure AWS env is set outside or here:
# os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

# -------------------- helpers --------------------
def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df.fillna("", inplace=True)
    logger.debug("Data loaded and NaNs filled from %s", file_path)
    return df

def load_model(model_path: str):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    logger.debug("Model loaded from %s", model_path)
    return model

def load_vectorizer(vectorizer_path: str) -> TfidfVectorizer:
    with open(vectorizer_path, "rb") as f:
        vec = pickle.load(f)
    logger.debug("TF-IDF vectorizer loaded from %s", vectorizer_path)
    return vec

def load_params(params_path: str) -> dict:
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    logger.debug("Parameters loaded from %s", params_path)
    return params

def evaluate_model(model, X_test, y_test) -> Tuple[dict, np.ndarray]:
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    logger.debug("Model evaluation completed")
    return report, cm

def log_confusion_matrix(cm, dataset_name: str):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix for {dataset_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    path = f"confusion_matrix_{dataset_name}.png"
    plt.savefig(path)
    plt.close()
    mlflow.log_artifact(path)

def ensure_experiment(tracking_uri: str, name: str, artifact_root: str) -> str:
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    exp = client.get_experiment_by_name(name)
    if exp is None:
        logger.info("Experiment '%s' does not exist. Creating with artifact root: %s", name, artifact_root)
        exp_id = client.create_experiment(name=name, artifact_location=artifact_root)
        return exp_id
    return exp.experiment_id

def write_experiment_info(run_id: str, model_subpath: str, params: dict, metrics: dict,
                          tracking_uri: str, experiment_name: str, artifact_root: str,
                          mlflow_error: str = None):
    payload = {
        "run_id": run_id,
        "model_path": model_subpath,
        "tracking_uri": tracking_uri,
        "experiment_name": experiment_name,
        "artifact_root": artifact_root,
        "used_mlflow": mlflow_error is None,
        "mlflow_error": mlflow_error,
        "params": params,
        "metrics": metrics,
    }
    with open("experiment_info.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.debug("Model/experiment info saved to experiment_info.json")

# -------------------- main --------------------
def main():
    used_mlflow_error = None
    run_id = None
    model_subpath = "lgbm_model"
    metrics_for_file = {}
    params_for_file = {}

    try:
        # Project roots
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        params = load_params(os.path.join(root_dir, "params.yaml"))
        params_for_file = params

        # Load artifacts
        model = load_model(os.path.join(root_dir, "lgbm_model.pkl"))
        vectorizer = load_vectorizer(os.path.join(root_dir, "tfidf_vectorizer.pkl"))
        test_df = load_data(os.path.join(root_dir, "data/interim/test_processed.csv"))

        X_test = vectorizer.transform(test_df["clean_comment"].values)
        y_test = test_df["category"].values

        # Prepare example/signature
        input_example = pd.DataFrame(
            X_test.toarray()[:5], columns=vectorizer.get_feature_names_out()
        )
        signature = infer_signature(input_example, model.predict(X_test[:5]))

        # Ensure experiment exists with correct artifact root
        exp_id = ensure_experiment(TRACKING_URI, EXPERIMENT_NAME, ARTIFACT_ROOT)
        mlflow.set_experiment(EXPERIMENT_NAME)

        # ---- MLflow logging (guarded) ----
        with mlflow.start_run() as run:
            run_id = run.info.run_id

            # Log params (flatten top-level only; adjust as needed)
            for k, v in params.items():
                mlflow.log_param(k, v)

            # Log model with signature
            mlflow.sklearn.log_model(
                model,
                model_subpath,
                signature=signature,
                input_example=input_example
            )

            # Log the vectorizer file
            mlflow.log_artifact(os.path.join(root_dir, "tfidf_vectorizer.pkl"))

            # Evaluate and log metrics
            report, cm = evaluate_model(model, X_test, y_test)

            # Aggregate a few overall metrics for the JSON file
            if "weighted avg" in report:
                metrics_for_file = {
                    "test_weighted_precision": report["weighted avg"]["precision"],
                    "test_weighted_recall": report["weighted avg"]["recall"],
                    "test_weighted_f1": report["weighted avg"]["f1-score"],
                    "test_accuracy": report.get("accuracy", None),
                }

            # Log all per-label metrics
            for label, m in report.items():
                if isinstance(m, dict) and {"precision","recall","f1-score"} <= m.keys():
                    mlflow.log_metrics({
                        f"test_{label}_precision": m["precision"],
                        f"test_{label}_recall": m["recall"],
                        f"test_{label}_f1": m["f1-score"],
                    })
            # Confusion matrix plot
            log_confusion_matrix(cm, "Test_Data")

    except Exception as e:
        used_mlflow_error = str(e)
        logger.error("Failed to complete model evaluation: %s", used_mlflow_error)
        print(f"Error: {used_mlflow_error}")

    # ---- Always write the file DVC expects ----
    write_experiment_info(
        run_id=run_id or "",
        model_subpath=model_subpath,
        params=params_for_file,
        metrics=metrics_for_file,
        tracking_uri=TRACKING_URI,
        experiment_name=EXPERIMENT_NAME,
        artifact_root=ARTIFACT_ROOT,
        mlflow_error=used_mlflow_error,
    )

if __name__ == "__main__":
    main()
