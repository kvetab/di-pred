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

# # Training functions

def knn(preprocessing, data_name, hp_dir):
    filename = path.join(hp_dir, f"kNN_{data_name}_{preprocessing}.json")
    parameters = json.load(open(filename))
    #n_neighbors = int(parameters["n_neighbors"])
    model = KNeighborsClassifier(n_neighbors=int(parameters["n_neighbors"]))
    return model, parameters, "kNN"

def logistic_regression(preprocessing, data_name, hp_dir):
    filename = path.join(hp_dir, f"logistic_regression_{data_name}_{preprocessing}.json")
    parameters = json.load(open(filename))
    #C = float(parameters["C"])
    lr = LogisticRegression(
        class_weight='balanced', max_iter=1000, random_state=42,
        C=float(parameters["C"]), penalty=parameters["penalty"], solver=parameters["solver"]
    )
    return lr, parameters, "logistic_regression"

def random_forest(preprocessing, data_name, hp_dir):
    filename = path.join(hp_dir, f"random_forest_{data_name}_{preprocessing}.json")
    parameters = json.load(open(filename))
    rf = RandomForestClassifier(
        random_state=42, n_jobs=-1, class_weight='balanced', n_estimators=int(parameters["n_estimators"]),
        max_depth=int(parameters["max_depth"]), max_features=float(parameters["max_features"])
    )
    return rf, parameters, "random_forest"

def multilayer_perceptron(preprocessing, data_name, hp_dir):
    filename = path.join(hp_dir, f"multilayer_perceptron_{data_name}_{preprocessing}.json")
    parameters = json.load(open(filename))
    mlp = MLPClassifier(
        random_state=42, max_iter=int(1000), hidden_layer_sizes=parameters["hidden_layer_sizes"],
        activation=parameters["activation"]
    )
    return mlp, parameters, "multilayer_perceptron"

def svm(preprocessing, data_name, hp_dir):
    filename = path.join(hp_dir, f"SVM_{data_name}_{preprocessing}.json")
    parameters = json.load(open(filename))
    svc = SVC(
        max_iter=8000, probability=True, class_weight='balanced', C=float(parameters["C"]),
        kernel=parameters["kernel"], gamma=float(parameters["gamma"])
    )
    return svc, parameters, "SVM"

def gradient_boosting(preprocessing, data_name, hp_dir):
    filename = path.join(hp_dir, f"gradient_boosting_{data_name}_{preprocessing}.json")
    parameters = json.load(open(filename))
    gb = GradientBoostingClassifier(
        random_state=42, n_iter_no_change=70, learning_rate=float(parameters["learning_rate"]),
        n_estimators=int(parameters["n_estimators"]), max_depth=int(parameters["max_depth"]),
        max_features=float(parameters["max_features"])
    )
    return gb, parameters, "gradient_boosting"

## Wrappers

def output_evaluation(model_type, metrics, data, outpath, preprocessing):
    filename_sum = os.path.join(DATA_DIR, f"evaluations/{outpath}/all.csv")
    #print(filename_sum)
    line = [model_type, data, preprocessing, metrics["f1"], metrics["mcc"], metrics["acc"],metrics["precision"],metrics["recall"],metrics["auc"]]
    with open(filename_sum, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter='\t')
        csvwriter.writerow(line)


def train_and_eval(model_name, classifier, X_train, y_train, X_valid, y_valid,
                   data_name, outpath, preprocessing):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_valid)
    filename = path.join(DATA_DIR, "evaluations", outpath, "models", f"{model_name}_{data_name}_{preprocessing}.pkl")
    with open(filename, 'wb') as f:
        pickle.dump(classifier, f)
    filename = path.join(DATA_DIR, "evaluations", outpath, f"{model_name}_{data_name}_{preprocessing}.csv")
    str_preds = [str(int(pred)) for pred in y_pred]
    with open(filename, "wt") as f:
        f.write(",".join(str_preds) + "\n")
    metric_dict = {
        "f1": float(metrics.f1_score(y_valid, y_pred)),
        "acc": float(metrics.accuracy_score(y_valid, y_pred)),
        "mcc": float(metrics.matthews_corrcoef(y_valid, y_pred)),
        "auc": float(metrics.roc_auc_score(y_valid, y_pred)),
        "precision": float(metrics.precision_score(y_valid, y_pred)),
        "recall": float(metrics.recall_score(y_valid, y_pred))
    }
    
    output_evaluation(model_name, metric_dict, data_name, outpath, preprocessing)


def test_on_tap(model_name, x_test, y_test,
                   data_name, outpath, preprocessing):
    filename = path.join(DATA_DIR, "evaluations", outpath, "models", f"{model_name}_{data_name}_{preprocessing}.pkl")
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
    line = [model_name, data_name, preprocessing, metric_dict["f1"], metric_dict["mcc"], metric_dict["acc"],metric_dict["precision"],metric_dict["recall"],metric_dict["auc"], filename]
    with open(filename_sum, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter='\t')
        csvwriter.writerow(line)
    filename = path.join(DATA_DIR, "evaluations", outpath, f"{model_name}_{data_name}_{preprocessing}_tap.csv")
    str_preds = [str(int(pred)) for pred in y_pred]
    with open(filename, "wt") as f:
        f.write(",".join(str_preds) + "\n")


# ## Loading data representations

def integer_encoded(train_df, test_df, tap_df):
    x_chen = pd.read_csv(path.join(DATA_DIR, "chen/integer_encoding/chen_integer_encoded.csv"), index_col=0)
    x_chen_train = x_chen.merge(train_df[["Antibody_ID", "Y"]].reset_index(), left_on="Ab_ID", right_on="Antibody_ID").set_index('index').drop("Antibody_ID", axis=1)
    x_chen_test = x_chen.merge(test_df[["Antibody_ID", "Y"]].reset_index(), left_on="Ab_ID", right_on="Antibody_ID").set_index('index').drop("Antibody_ID", axis=1)
    x_tap = pd.read_csv(path.join(DATA_DIR, "tap/integer_encoding/tap_integer_encoded.csv"))
    x_tap.drop("Ab_ID", axis=1, inplace=True)
    x_tap = x_tap.loc[tap_df.index]
    return x_chen_train, x_chen_test, x_tap


def pybiomed(train_df, test_df, tap_df):
    x_chen = pd.read_feather(path.join(DATA_DIR, "chen/pybiomed/X_data.ftr"))
    x_chen_train = x_chen.merge(train_df[["Antibody_ID", "Y"]].reset_index(), left_on="Ab_ID", right_on="Antibody_ID").set_index('index').drop("Antibody_ID", axis=1)
    x_chen_test = x_chen.merge(test_df[["Antibody_ID", "Y"]].reset_index(), left_on="Ab_ID", right_on="Antibody_ID").set_index('index').drop("Antibody_ID", axis=1)
    x_tap = pd.read_feather(path.join(DATA_DIR, "tap/pybiomed/X_TAP_data.ftr"))
    x_tap = x_tap.loc[tap_df.index]
    return x_chen_train, x_chen_test, x_tap


def protparam(train_df, test_df, tap_df):
    x_chen = pd.read_csv(path.join(DATA_DIR, "chen/protparam/protparam_features.csv"))
    x_chen.rename({"Unnamed: 0": "Ab_ID"}, axis=1, inplace=True)
    x_chen = x_chen.drop("name", axis=1)
    x_chen_train = x_chen.merge(train_df[["Antibody_ID", "Y"]].reset_index(), left_on="Ab_ID", right_on="Antibody_ID").set_index('index').drop("Antibody_ID", axis=1)
    x_chen_test = x_chen.merge(test_df[["Antibody_ID", "Y"]].reset_index(), left_on="Ab_ID", right_on="Antibody_ID").set_index('index').drop("Antibody_ID", axis=1)
    
    x_tap = pd.read_csv(path.join(DATA_DIR, "tap/protparam/protparam_features_tap.csv"))
    x_tap = x_tap.merge(tap_df[["Antibody_ID", "Y"]].reset_index(), right_on="Antibody_ID", left_on="Unnamed: 0").set_index('index').drop("Antibody_ID", axis=1)
    x_tap = x_tap.drop(["Unnamed: 0", "Y"], axis=1)
    return x_chen_train, x_chen_test, x_tap


def bert(train_df, test_df, tap_df):
    x_chen = pd.read_feather(path.join(DATA_DIR, "chen/embeddings/bert/bert_chen_embeddings.ftr"))
    x_chen_train = x_chen.merge(train_df[["Antibody_ID", "Y"]].reset_index(), left_on="Ab_ID", right_on="Antibody_ID").set_index('index').drop("Antibody_ID", axis=1)
    x_chen_test = x_chen.merge(test_df[["Antibody_ID", "Y"]].reset_index(), left_on="Ab_ID", right_on="Antibody_ID").set_index('index').drop("Antibody_ID", axis=1)
    x_tap = pd.read_feather(path.join(DATA_DIR, "tap/embeddings/bert/bert_tap_embeddings.ftr"))
    x_tap = x_tap.drop("Ab_ID", axis=1)
    x_tap = x_tap.loc[tap_df.index]
    return x_chen_train, x_chen_test, x_tap


def seqvec(train_df, test_df, tap_df):
    x_chen = pd.read_feather(path.join(DATA_DIR, "chen/embeddings/seqvec/seqvec_chen_embeddings.ftr"))
    x_chen_train = x_chen.merge(train_df[["Antibody_ID", "Y"]].reset_index(), left_on="Ab_ID", right_on="Antibody_ID").set_index('index').drop("Antibody_ID", axis=1)
    x_chen_test = x_chen.merge(test_df[["Antibody_ID", "Y"]].reset_index(), left_on="Ab_ID", right_on="Antibody_ID").set_index('index').drop("Antibody_ID", axis=1)
    x_tap = pd.read_csv(path.join(DATA_DIR, "tap/embeddings/seqvec/seqvec_tap_embeddings.csv"), index_col=0)
    x_tap = x_tap.drop("Ab_ID", axis=1)
    x_tap = x_tap.loc[tap_df.index]
    return x_chen_train, x_chen_test, x_tap


def sapiens(train_df, test_df, tap_df):
    x_chen = pd.read_csv(path.join(DATA_DIR, "chen/embeddings/sapiens/sapiens_chen_embeddings.csv"), index_col=0).drop("Y", axis=1)
    x_chen_train = x_chen.merge(train_df[["Antibody_ID", "Y"]].reset_index(), left_on="Ab_ID", right_on="Antibody_ID").set_index('index').drop("Antibody_ID", axis=1)
    x_chen_test = x_chen.merge(test_df[["Antibody_ID", "Y"]].reset_index(), left_on="Ab_ID", right_on="Antibody_ID").set_index('index').drop("Antibody_ID", axis=1)
    x_tap = pd.read_csv(path.join(DATA_DIR, "tap/embeddings/sapiens/sapiens_tap_embeddings.csv"), index_col=0)
    x_tap = x_tap.drop(["Ab_ID", "Y"], axis=1)
    x_tap = x_tap.loc[tap_df.index]
    return x_chen_train, x_chen_test, x_tap


def onehot(train_df, test_df, tap_df):
    x_chen = pd.read_feather(path.join(DATA_DIR, "chen/onehot/chen_onehot.ftr"))
    x_chen_train = x_chen.merge(train_df[["Antibody_ID", "Y"]].reset_index(), left_on="Ab_ID", right_on="Antibody_ID").set_index('index').drop("Antibody_ID", axis=1)
    x_chen_test = x_chen.merge(test_df[["Antibody_ID", "Y"]].reset_index(), left_on="Ab_ID", right_on="Antibody_ID").set_index('index').drop("Antibody_ID", axis=1)
    x_tap = pd.read_feather(path.join(DATA_DIR, "tap/onehot/tap_onehot.ftr"))
    x_tap = x_tap.drop(["Ab_ID"], axis=1)
    x_tap = x_tap.loc[tap_df.index]
    return x_chen_train, x_chen_test, x_tap


# ## Preprocessing

def no_prepro(train_df, test_df, tap_df):
    return train_df.drop("Y", axis=1), train_df["Y"], test_df, tap_df


def scaling(train_df, test_df, tap_df):
    scaler = StandardScaler()
    scaler.fit(train_df.drop(["Ab_ID", "Y"], axis=1))
    x_train_tr = scaler.transform(train_df.drop(["Ab_ID", "Y"], axis=1))
    x_train_df = pd.DataFrame(data=train_df,  index=train_df.index, columns=train_df.drop(["Ab_ID", "Y"], axis=1).columns)
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
    default=4
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

# parse the command line
args = CLI.parse_args()

data_loaders = [data_loaders[j] for j in args.data]
model_creators = [model_creators[j] for j in args.models]
preprocessing = [preprocessing[j] for j in args.prepro]
seed = args.seed



def crossval_round(train_df, test_df, eval_dir, tap_data):
    for data_rep in data_loaders:
        data_name = data_rep.__name__
        x_train, x_test, x_tap = data_rep(train_df, test_df, tap_data)
        for prepro in preprocessing:
            prepro_name = prepro.__name__
            x_train_tr, y_train_tr, x_test_tr, tap_tr = prepro(x_train, x_test, x_tap)
            for model_creator in model_creators:
                classifier, params, model_label = model_creator(prepro_name, data_name, path.join(DATA_DIR, "hyperparameters"))
                train_and_eval(
                    model_label, classifier, x_train_tr.drop(["Ab_ID"], axis=1), 
                    y_train_tr, x_test_tr.drop(["Ab_ID", "Y"], axis=1), x_test_tr["Y"], 
                    data_name, eval_dir, prepro_name
                )
                test_on_tap(
                    model_label, tap_tr, tap_data["Y"], data_name, eval_dir, prepro_name
                )
            

chen_train = pd.read_csv(path.join(DATA_DIR, f"chen/deduplicated/crossval/chen_{seed}_a.csv"), index_col=0).drop("cluster", axis=1).sort_index()
chen_test = pd.read_csv(path.join(DATA_DIR, f"chen/deduplicated/crossval/chen_{seed}_b.csv"), index_col=0).drop("cluster", axis=1).sort_index()
tap_data = pd.read_csv(path.join(DATA_DIR, "tap/tap_not_in_chen.csv"), index_col=0)


eval_dir = f"training_split_{seed}_a"

os.mkdir(os.path.join(DATA_DIR, f"evaluations/{eval_dir}"))
os.mkdir(os.path.join(DATA_DIR, f"evaluations/{eval_dir}/models"))
crossval_round(chen_train, chen_test, eval_dir, tap_data)

eval_dir = f"training_split_{seed}_b"
os.mkdir(os.path.join(DATA_DIR, f"evaluations/{eval_dir}"))
os.mkdir(os.path.join(DATA_DIR, f"evaluations/{eval_dir}/models"))
crossval_round(chen_test, chen_train, eval_dir, tap_data)

