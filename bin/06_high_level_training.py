#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from os import path
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.utils.fixes import loguniform    
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneGroupOut
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from collections import Counter
import pickle
import numpy as np
import os
import json
import csv
import sys
import argparse

import warnings
warnings.filterwarnings("ignore")


DATA_DIR = "."

# # Load Data

chen_data = pd.read_csv(path.join(DATA_DIR, "chen/deduplicated/chen_data.csv"), index_col=0)
tap_data = pd.read_csv(path.join(DATA_DIR, "tap/TAP_data.csv"))

clusters = pd.read_csv(path.join(DATA_DIR, "chen/clustering.csv"), index_col=0)

seeds = [2, 13, 19, 27, 38, 42, 56, 63, 6, 78]



# # Training functions

def merge_clusters(train_df, cluster_df):
    df = train_df.merge(cluster_df, left_index=True, right_index=True).rename({"0": "cluster"}, axis=1)
    df["cluster_merged"] = df["cluster"]
    df["cluster_merged"][df["cluster"] < 300] = df["cluster"][df["cluster"] < 300] // 30
    df["cluster_merged"][df["cluster"] >= 300] = df["cluster"][df["cluster"] >= 300] // 100
    print(f'Unique clusters after merge: {df["cluster_merged"].nunique()}')
    return df

def knn(n):
    model = KNeighborsClassifier()  # default metric is Euclidean
    parameters = {'n_neighbors': [1,3,5]}
    return model, parameters, "kNN"

def logistic_regression(n):
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    parameters = {'C':loguniform(0.001, 1000), 'penalty': ["l2"], "solver": ["lbfgs", "sag"]}
    return lr, parameters, "logistic_regression"

def random_forest(n):
    rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
    parameters = {'n_estimators': np.arange(1, 200, 10), 'max_depth': np.arange(1, min(50,n), 2), 
                  'max_features': np.arange(0.1, 0.75, 0.05)}
    return rf, parameters, "random_forest"

def multilayer_perceptron(n):
    mlp = MLPClassifier(random_state=42, max_iter=int(1000))
    parameters = {'hidden_layer_sizes': [(100,), (50,), (50, 50), (100, 100)], "activation": ["relu", "logistic"]}
    return mlp, parameters, "multilayer_perceptron"

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

## Wrappers

def output_evaluation(model_type, params, best_params, metrics, data, outpath, preprocessing):
    prepro = "_"+preprocessing if preprocessing is not None else ""
    filename = os.path.join(DATA_DIR, "evaluations", outpath, f"{model_type}_{data}{prepro}.json")
    out_dict = {
        "model_type": model_type,
        "data": data
    }
    out_dict["params"] = {}
    out_dict["best_params"] = {}
    for key, value in params.items():
        out_dict["params"][key] = str(value)
    for key, value in best_params.items():
        out_dict["best_params"][key] = str(value)
    out_dict["metrics"] = metrics
    out_dict["preprocessing"] = "none" if preprocessing is None else preprocessing
    
    json.dump(out_dict, open(filename, "w"))
    
    filename_sum = os.path.join(DATA_DIR, f"evaluations/{outpath}/all.csv")
    #print(filename_sum)
    line = [model_type, data, out_dict["preprocessing"], metrics["f1"], metrics["mcc"], metrics["acc"],metrics["precision"],metrics["recall"],metrics["auc"], filename]
    with open(filename_sum, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter='\t')
        csvwriter.writerow(line)


def train_and_eval(model_name, classifier, parameters, X_train, y_train, X_valid, y_valid, groups,
                   data_name, outpath, preprocessing):
    splitter = LeaveOneGroupOut()
    split = splitter.split(X_train, y_train, groups=groups)
    grid = RandomizedSearchCV(classifier, parameters, verbose=0, scoring="f1", cv=split)
    grid.fit(X_train, y_train)
    estimator = grid.best_estimator_
    best_params = grid.best_params_
    y_pred = estimator.predict(X_valid)
    filename = path.join(DATA_DIR, "evaluations", outpath, "models", f"{model_name}_{data_name}_{preprocessing}.pkl")
    with open(filename, 'wb') as f:
        pickle.dump(estimator, f)
    metric_dict = {
        "f1": float(metrics.f1_score(y_valid, y_pred)),
        "acc": float(metrics.accuracy_score(y_valid, y_pred)),
        "mcc": float(metrics.matthews_corrcoef(y_valid, y_pred)),
        "auc": float(metrics.roc_auc_score(y_valid, y_pred)),
        "precision": float(metrics.precision_score(y_valid, y_pred)),
        "recall": float(metrics.recall_score(y_valid, y_pred))
    }
    
    print(f"{model_name}, {data_name}, {preprocessing}")
    print(f"F1: {metric_dict['f1']}")
    print(f"MCC: {metric_dict['mcc']}")
    print(f"Accuracy: {metric_dict['acc']}")
    print(f"Precision: {metric_dict['precision']}")
    print(f"Recall: {metric_dict['recall']}")
    print(f"AUC: {metric_dict['auc']}")
    print(f"-----")
    
    output_evaluation(model_name, parameters, best_params, metric_dict, data_name, outpath, preprocessing)


def try_all(X_train, y_train, X_valid, y_valid, groups, data_name, outpath, preprocessing, models):
    n = len(y_train)
    for model_creator in models:
    #for model_creator in [logistic_regression, random_forest]:

        classifier, params, model_label = model_creator(n)
        print("\n")
        print(f'Training model {model_label} on data {data_name} with preprocessing {preprocessing} \n')
        train_and_eval(model_label, classifier, params, X_train, y_train, X_valid, y_valid, groups,
                   data_name, outpath, preprocessing)


def test_on_tap(model_name, x_test, y_test,
                   data_name, outpath, preprocessing=None):
    prepro = "_"+preprocessing if preprocessing is not None else ""
    filename = path.join(DATA_DIR, "evaluations", outpath, "models", f"{model_name}_{data_name}{prepro}.pkl")
    with open(filename, 'rb') as f:
        estimator = pickle.load(f)
    y_pred = estimator.predict(x_test)
    metric_dict = {
        "f1": float(metrics.f1_score(y_test, y_pred)),
        "acc": float(metrics.accuracy_score(y_test, y_pred)),
        "mcc": float(metrics.matthews_corrcoef(y_test, y_pred)),
        "auc": float(metrics.roc_auc_score(y_test, y_pred)),
        "precision": float(metrics.precision_score(y_test, y_pred)),
        "recall": float(metrics.recall_score(y_test, y_pred))
    }
    filename_sum = os.path.join(DATA_DIR, f"evaluations/{outpath}/tap.csv")
    line = [model_name, data_name, prepro, metric_dict["f1"], metric_dict["mcc"], metric_dict["acc"],metric_dict["precision"],metric_dict["recall"],metric_dict["auc"], filename]
    with open(filename_sum, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter='\t')
        csvwriter.writerow(line)


def test_all(x_test, y_test, data_name, outpath, preprocessing, models):
    for model in models:
        print(f"Testing model {model} on {data_name} with preprocessing {preprocessing}...")
        test_on_tap(model, x_test, y_test, data_name, outpath, preprocessing=preprocessing)


# ## Loading data representations

def integer_encoded(train_df, test_df):
    x_chen = pd.read_csv(path.join(DATA_DIR, "chen/integer_encoding/chen_integer_encoded.csv"), index_col=0)
    x_chen_train = x_chen.merge(train_df[["Antibody_ID", "Y", "cluster_merged"]], left_on="Ab_ID", right_on="Antibody_ID").drop("Antibody_ID", axis=1)
    x_chen_test = x_chen.merge(test_df[["Antibody_ID", "Y"]], left_on="Ab_ID", right_on="Antibody_ID").drop("Antibody_ID", axis=1)
    x_tap = pd.read_csv(path.join(DATA_DIR, "tap/integer_encoding/tap_integer_encoded.csv"))
    x_tap.drop("Ab_ID", axis=1, inplace=True)
    return x_chen_train, x_chen_test, x_tap


def pybiomed(train_df, test_df):
    x_chen = pd.read_feather(path.join(DATA_DIR, "chen/pybiomed/X_data.ftr"))
    x_chen_train = x_chen.merge(train_df[["Antibody_ID", "Y", "cluster_merged"]], left_on="Ab_ID", right_on="Antibody_ID").drop("Antibody_ID", axis=1)
    x_chen_test = x_chen.merge(test_df[["Antibody_ID", "Y"]], left_on="Ab_ID", right_on="Antibody_ID").drop("Antibody_ID", axis=1)
    x_tap = pd.read_feather(path.join(DATA_DIR, "tap/pybiomed/X_TAP_data.ftr"))
    return x_chen_train, x_chen_test, x_tap


def protparam(train_df, test_df):
    x_chen = pd.read_csv(path.join(DATA_DIR, "chen/protparam/protparam_features.csv"))
    x_chen.rename({"Unnamed: 0": "Ab_ID"}, axis=1, inplace=True)
    x_chen = x_chen.drop("name", axis=1)
    x_chen_train = x_chen.merge(train_df[["Antibody_ID", "Y", "cluster_merged"]], left_on="Ab_ID", right_on="Antibody_ID").drop("Antibody_ID", axis=1)
    x_chen_test = x_chen.merge(test_df[["Antibody_ID", "Y"]], left_on="Ab_ID", right_on="Antibody_ID").drop("Antibody_ID", axis=1)
    
    x_tap = pd.read_csv(path.join(DATA_DIR, "tap/protparam/protparam_features_tap.csv"))
    x_tap = x_tap.drop("Unnamed: 0", axis=1)
    return x_chen_train, x_chen_test, x_tap


def bert(train_df, test_df):
    x_chen = pd.read_feather(path.join(DATA_DIR, "chen/embeddings/bert/bert_chen_embeddings.ftr"))
    x_chen_train = x_chen.merge(train_df[["Antibody_ID", "Y", "cluster_merged"]], left_on="Ab_ID", right_on="Antibody_ID").drop("Antibody_ID", axis=1)
    x_chen_test = x_chen.merge(test_df[["Antibody_ID", "Y"]], left_on="Ab_ID", right_on="Antibody_ID").drop("Antibody_ID", axis=1)
    x_tap = pd.read_feather(path.join(DATA_DIR, "tap/embeddings/bert/bert_tap_embeddings.ftr"))
    x_tap = x_tap.drop("Ab_ID", axis=1)
    return x_chen_train, x_chen_test, x_tap


def seqvec(train_df, test_df):
    x_chen = pd.read_feather(path.join(DATA_DIR, "chen/embeddings/seqvec/seqvec_chen_embeddings.ftr"))
    x_chen_train = x_chen.merge(train_df[["Antibody_ID", "Y", "cluster_merged"]], left_on="Ab_ID", right_on="Antibody_ID").drop("Antibody_ID", axis=1)
    x_chen_test = x_chen.merge(test_df[["Antibody_ID", "Y"]], left_on="Ab_ID", right_on="Antibody_ID").drop("Antibody_ID", axis=1)
    x_tap = pd.read_feather(path.join(DATA_DIR, "tap/embeddings/seqvec/seqvec_tap_embeddings.ftr"))
    x_tap = x_tap.drop("Ab_ID", axis=1)
    return x_chen_train, x_chen_test, x_tap


def sapiens(train_df, test_df):
    x_chen = pd.read_feather(path.join(DATA_DIR, "chen/embeddings/sapiens/sapiens_chen_embeddings.ftr")).drop("Y", axis=1)
    x_chen_train = x_chen.merge(train_df[["Antibody_ID", "Y", "cluster_merged"]], left_on="Ab_ID", right_on="Antibody_ID").drop("Antibody_ID", axis=1)
    x_chen_test = x_chen.merge(test_df[["Antibody_ID", "Y"]], left_on="Ab_ID", right_on="Antibody_ID").drop("Antibody_ID", axis=1)
    x_tap = pd.read_feather(path.join(DATA_DIR, "tap/embeddings/sapiens/sapiens_tap_embeddings.ftr"))
    x_tap = x_tap.drop(["Ab_ID", "Y"], axis=1)
    return x_chen_train, x_chen_test, x_tap


def onehot(train_df, test_df):
    x_chen = pd.read_feather(path.join(DATA_DIR, "chen/onehot/chen_onehot.ftr"))
    x_chen_train = x_chen.merge(train_df[["Antibody_ID", "Y", "cluster_merged"]], left_on="Ab_ID", right_on="Antibody_ID").drop("Antibody_ID", axis=1)
    x_chen_test = x_chen.merge(test_df[["Antibody_ID", "Y"]], left_on="Ab_ID", right_on="Antibody_ID").drop("Antibody_ID", axis=1)
    x_tap = pd.read_feather(path.join(DATA_DIR, "tap/onehot/tap_onehot.ftr"))
    x_tap = x_tap.drop(["Ab_ID"], axis=1)
    return x_chen_train, x_chen_test, x_tap


# ## Preprocessing

def no_prepro(train_df, test_df, tap_df):
    return train_df.drop("Y", axis=1), train_df["Y"], test_df, tap_df


def scaling(train_df, test_df, tap_df):
    scaler = StandardScaler()
    scaler.fit(train_df.drop(["Ab_ID", "Y", "cluster_merged"], axis=1))
    x_train_tr = scaler.transform(train_df.drop(["Ab_ID", "Y", "cluster_merged"], axis=1))
    x_train_df = pd.DataFrame(data=train_df,  index=train_df.index, columns=train_df.drop(["Ab_ID", "Y", "cluster_merged"], axis=1).columns)
    x_train_df["cluster_merged"] = train_df["cluster_merged"]
    x_train_df["Ab_ID"] = train_df["Ab_ID"]
    
    x_test_tr = scaler.transform(test_df.drop(["Ab_ID", "Y"], axis=1))
    x_test_df = pd.DataFrame(data=test_df,  index=test_df.index, columns=test_df.drop(["Ab_ID", "Y"], axis=1).columns)
    x_test_df["Y"] = test_df["Y"]
    x_test_df["Ab_ID"] = test_df["Ab_ID"]
    
    x_tap_tr = scaler.transform(tap_df)
    x_tap_df = pd.DataFrame(data=tap_df,  index=tap_df.index, columns=tap_df.columns)

    return x_train_df, train_df["Y"], x_test_df, x_tap_df


def oversampling(train_df, test_df, tap_df):
    sampler = RandomOverSampler(random_state=42)
    x_train, y_train = sampler.fit_resample(train_df.drop("Y", axis=1), train_df["Y"])
    return x_train, y_train, test_df, tap_df


def smote_os(train_df, test_df, tap_df):
    sampler = SMOTE(random_state=42)
    x_train_tr, y_train = sampler.fit_resample(train_df.drop(["Ab_ID", "Y"], axis=1), train_df["Y"])
    x_train_tr["Ab_ID"] = ""
    return x_train_tr, y_train, test_df, tap_df


def undersampling(train_df, test_df, tap_df):
    sampler = RandomUnderSampler(random_state=42)
    x_train, y_train = sampler.fit_resample(train_df.drop("Y", axis=1), train_df["Y"]) 
    return x_train, y_train, test_df, tap_df


# ## All together


data_loaders = [integer_encoded, pybiomed, protparam, bert, seqvec, sapiens, onehot]
model_creators = [knn, logistic_regression, random_forest, multilayer_perceptron, svm, gradient_boosting]
preprocessing = [no_prepro, scaling, oversampling, smote_os, undersampling]


# Input from console


CLI=argparse.ArgumentParser()
CLI.add_argument(
    "--seed",
    type=int,
    default=2
)
CLI.add_argument(
    "--data",  # name on the CLI - drop the `--` for positional/required parameters
    nargs="+",  # 0 or more values expected => creates a list
    type=int,
    default=[0,1,2,3,4,5,6],  # default if nothing is provided
)
CLI.add_argument(
    "--models",
    nargs="+",
    type=int,  # any type/callable can be used here
    default=[0,1,2,3,4,5],
)
CLI.add_argument(
    "--prepro",
    nargs="+",
    type=int,  
    default=[0,1,2,3,4],
)
CLI.add_argument(
    "--filter",
    type=int,
    default=0
)

# parse the command line
args = CLI.parse_args()

data_loaders = [data_loaders[j] for j in args.data]
model_creators = [model_creators[j] for j in args.models]
preprocessing = [preprocessing[j] for j in args.prepro]
seed = args.seed



def crossval_round(train_df, test_df, eval_dir):
    for prepro in preprocessing:
        prepro_name = prepro.__name__
        for data_rep in data_loaders:
            data_name = data_rep.__name__
            x_train, x_test, x_tap = data_rep(train_df, test_df)
            x_train_tr, y_train_tr, x_test_tr, tap_tr = prepro(x_train, x_test, x_tap)
            
            try_all(x_train_tr.drop(["Ab_ID", "cluster_merged"], axis=1), y_train_tr, 
                x_test_tr.drop(["Ab_ID", "Y"], axis=1), x_test_tr["Y"], 
                x_train_tr["cluster_merged"], data_name, eval_dir, prepro_name, model_creators)
            
            #test_all(tap_tr, tap_data["Y"], data_name, eval_dir, prepro_name, model_creators)



chen_filtered = pd.read_csv(path.join(DATA_DIR, f"chen/deduplicated/filtered_chen_data.csv"), index_col=0)
chen_train = pd.read_csv(path.join(DATA_DIR, f"chen/deduplicated/crossval/chen_train_{seed}.csv"), index_col=0)
#chen_train = chen_train[:50]
chen_train = merge_clusters(chen_train, clusters)
if args.filter > 0:
    chen_train = chen_train.merge(chen_filtered[["Antibody_ID"]], on="Antibody_ID")

chen_test = pd.read_csv(path.join(DATA_DIR, f"chen/deduplicated/crossval/chen_test_{seed}.csv"), index_col=0)
if args.filter > 0:
    chen_test = chen_test.merge(chen_filtered[["Antibody_ID"]], on="Antibody_ID")
eval_dir = f"training_split_{seed}"
if args.filter > 0:
    eval_dir += "_filt"
#try:
os.mkdir(os.path.join(DATA_DIR, f"evaluations/{eval_dir}"))
os.mkdir(os.path.join(DATA_DIR, f"evaluations/{eval_dir}/models"))
#except:
#    pass
crossval_round(chen_train, chen_test, eval_dir)




