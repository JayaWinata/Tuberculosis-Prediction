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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

params = yaml.safe_load(open('params.yaml'))['train']

def evaluate(data_path, model_path):
    data = pd.read_csv(data_path)
    X = data.drop(columns=['Class'])
    y = data['Class']

    model = pickle.load(open(model_path, 'rb'))

    prediction = model.predict(X)
    accuracy = accuracy_score(prediction, y)

    logging.info(f'Model accuracy: {accuracy}')

if __name__ == '__main__':
    input_map = {
        'v1': 0,
        'v2': 1,
        'v3': 2,
        'v4': 3,
        'v5': 4,
        'v6': 5
    }

    evaluate(params['input'][input_map['v1']]['path'], params['output'])
