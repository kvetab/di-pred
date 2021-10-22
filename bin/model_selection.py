import pandas as pd
import numpy as np
import json
import csv
import os
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.utils.fixes import loguniform    
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def logistic_regression(n):
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    parameters = {'C':loguniform(0.001, 1000), 'penalty': ["l2"], "solver": ["lbfgs", "sag"]}
    return lr, parameters, "logistic_regression"


def random_forest(n):
    rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
    parameters = {'n_estimators': np.arange(1, 200, 10), 'max_depth': np.arange(1, min(50,n), 2), 
                  'max_features': np.arange(0.1, 0.75, 0.05)}
    return rf, parameters, "random_forest"
    
    
def svm(n):
    svc = SVC(max_iter=8000, probability=True, class_weight='balanced')
    parameters = {'C': loguniform(0.001, 100), 'kernel':["linear", "rbf"], 'gamma': loguniform(1e-3, 1e0)}
    return svc, parameters, "SVM"


def gradient_boosting(n):
    gb = GradientBoostingClassifier(random_state=42, n_iter_no_change=70)
    parameters = {'learning_rate': loguniform(0.01, 0.5), 
                  'n_estimators': np.arange(1, 200, 10), 
                  'max_depth': np.arange(1, min(20,n), 2), 
                  'max_features': np.arange(0.1, 0.6, 0.1)}
    return gb, parameters, "gradient_boosting"


def multilayer_perceptron(n):
    mlp = MLPClassifier(random_state=42, max_iter=int(1000))
    parameters = {'hidden_layer_sizes': [(100,), (50,), (100, 100)], "activation": ["relu", "logistic"]}
    return mlp, parameters, "multilayer_perceptron"


def output_evaluation(model_type, params, metrics, data, preprocessing=None):
    prepro = "_"+preprocessing if preprocessing is not None else ""
    filename = f"../evaluations/{model_type}_{data}{prepro}"
    out_dict = {
        "model_type": model_type,
        "data": data
    }
    out_dict["params"] = {}
    for key, value in params.items():
        out_dict["params"]["key"] = str(value)
    out_dict["metrics"] = metrics
    out_dict["preprocessing"] = "none" if preprocessing is None else preprocessing
    
    json.dump(out_dict, open(filename, "w"))
    
    filename_sum = "../evaluations/all.csv"
    line = [model_type, data, metrics["f1"], metrics["mcc"], metrics["acc"], filename]
    with open(filename_sum, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter='\t')
        csvwriter.writerow(line)
    


def train_and_eval(model_name, classifier, parameters, X_train, y_train, X_valid, y_valid, 
                   data_name, preprocessing=None):
    grid = RandomizedSearchCV(classifier, parameters, verbose=2, scoring="f1")
    grid.fit(X_train, y_train)
    estimator = grid.best_estimator_
    best_params = grid.best_params_
    y_pred = estimator.predict(X_valid)
    metric_dict = {
        "f1": float(metrics.f1_score(y_valid, y_pred)),
        "acc": float(metrics.accuracy_score(y_valid, y_pred)),
        "mcc": float(metrics.matthews_corrcoef(y_valid, y_pred)),
        "auc": float(metrics.roc_auc_score(y_valid, y_pred)),
        "precision": float(metrics.precision_score(y_valid, y_pred)),
        "recall": float(metrics.recall_score(y_valid, y_pred))
    }
    output_evaluation(model_name, best_params, metric_dict, data_name, preprocessing)
    

EMBEDDINGS = [f"data/embeddings/{filename}" for filename in os.listdir("data/embeddings") if filename.startswith("embeddings_train")]
DATA = [{
    "X_train": "data/X_train_data.ftr",
    "y_train": "data/chen_train_data.csv",
    "X_valid": "data/X_valid_data.ftr",
    "y_valid": "data/chen_valid_data.csv",
    "label": "PyBioMed"
    
}]
DATA += [
    {
        "X_train": filename,
        "y_train": "data/chen_train_data.csv",
        "X_valid": filename.replace("train", "valid"),
        "y_valid": "data/chen_valid_data.csv",
        "label": os.path.basename(filename).replace("embeddings_train_", "").replace(".ftr", "")
    } for filename in EMBEDDINGS
]

MODELS = [logistic_regression, random_forest, svm, gradient_boosting, multilayer_perceptron]
    
    
        
if __name__ == "__main__":
    for data_dict in DATA:
        scaler = StandardScaler()
        # load data from ftr
        X_train = pd.read_feather(data_dict["X_train"]).drop("Ab_ID", axis=1)
        scaler.fit(X_train)
        X_scaled = scaler.transform(X_train)
        y_train = pd.read_csv(data_dict["y_train"], sep=";")["Y"]
        X_valid = pd.read_feather(data_dict["X_valid"]).drop("Ab_ID", axis=1)
        X_val_scaled = scaler.transform(X_valid)
        y_valid = pd.read_csv(data_dict["y_valid"], sep=";")["Y"]
        n = len(y_train)
        for model_creator in MODELS:
            classifier, params, model_label = model_creator(n)
            print("\n")
            print(f'Training model {model_label} on data {data_dict["label"]} \n')
            #train_and_eval(model_label, classifier, params, X_train, y_train, X_valid, y_valid, data_dict["label"])
            train_and_eval(model_label, classifier, params, X_scaled, y_train, X_val_scaled, y_valid, data_dict["label"], preprocessing="StandardScaler")
            