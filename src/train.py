from sklearn.metrics import make_scorer, accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pandas as pd
from sklearn.base import clone
import matplotlib.pyplot as plt

from mlflow.models import infer_signature
from urllib.parse import urlparse
import dagshub
import logging
import mlflow
import pickle
import yaml
import os
import time

dagshub.init(repo_owner='jayawinata100', repo_name='Tuberculosis-Prediction', mlflow=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

params = yaml.safe_load(open("params.yaml"))['train']

models = [
    {
        'model': LogisticRegression(),
        'name': 'Logistic Regression',
        'param_grid': {
            'penalty': ['l1', 'l2', 'elasticnet', None],
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['saga', 'liblinear'],
            'max_iter': [100, 200, 500]
        }
    },
    {
        'model': SVC(probability=True),
        'name': 'Support Vector Classifier',
        'param_grid': dict()
    },
    {
        'model': RandomForestClassifier(),
        'name': 'Random Forest Classifier',
        'param_grid': {
            'n_estimators': [100, 200, 500],
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }
    }
]

datasets = [
    [item['name'], pd.read_csv(item['path'])]
    for item in params['input']
]

def train(output_model_path):
    mlflow.set_experiment("Tuberculosis Prediction")
    # mlflow.set_tracking_uri("http://localhost:5000")
    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    total_runs = len(datasets) * len(models)
    run_counter = 1

    best_model = None
    best_score = 0.0
    best_model_name = ""
    best_signature = None

    with mlflow.start_run(run_name="Tuberculosis Prediction Model Training", nested=False):
        for i in datasets:
            dataset_name = i[0]
            logging.info(f"Starting training on dataset: {dataset_name}")

            X = i[-1].drop(columns=['Class'])
            y = i[-1]['Class']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            for j in models:
                run_name = f"{dataset_name} + {j['name']}"
                logging.info(f"[{run_counter}/{total_runs}] Training model: {run_name}")
                run_counter += 1

                start_time = time.time()
                base_model = clone(j['model'])
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                model = RandomizedSearchCV(base_model, j['param_grid'], cv=cv, random_state=42, scoring='accuracy', refit='accuracy')
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                training_time = time.time() - start_time

                if accuracy > best_score:
                    best_score = accuracy
                    best_model = model.best_estimator_
                    best_model_name = run_name
                    best_signature = infer_signature(X_train, y_train)

                signature = infer_signature(X_train, y_train)
                with mlflow.start_run(run_name=run_name, nested=True):
                    mlflow.set_tags({
                        "dataset": dataset_name,
                        "model_type": j['name'],
                    })
                    mlflow.log_param("model", run_name)
                    mlflow.log_params(model.best_params_)
                    mlflow.log_metric('accuracy', accuracy)
                    mlflow.log_metric("training_time", training_time)

                    cm = confusion_matrix(y_test, y_pred)
                    cr = classification_report(y_test, y_pred)

                    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                    fig, ax = plt.subplots(figsize=(6, 6))
                    disp.plot(ax=ax)
                    plt.title("Confusion Matrix")
                    mlflow.log_figure(fig, "confusion_matrix_plot.png")
                    plt.close(fig)

                    mlflow.log_text(str(cm), "confusion_matrix.txt")
                    mlflow.log_text(cr, "classification_report.txt")

                    if tracking_url_type_store != 'file':
                        mlflow.sklearn.log_model(model, "model", registered_model_name=f"Best {run_name}")
                    else:
                        mlflow.sklearn.log_model(model, "model", signature=signature)

                    logging.info(f"✅ Finished training: {run_name} | Accuracy: {accuracy:.4f}")

    mlflow.sklearn.log_model(
                        sk_model=best_model,
                        artifact_path="model",
                        registered_model_name=run_name
    )

    if best_model:
        pickle.dump(best_model, open(output_model_path, 'wb'))
        logging.info(f"Best model '{best_model_name}' saved with accuracy: {best_score:.4f}")
    else:
        logging.warning("No model was selected as best.")

if __name__ == '__main__':
    train(params['output'])