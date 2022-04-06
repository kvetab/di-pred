#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from os import path
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
import numpy as np
import os
import json
import csv
import sys
import argparse

import warnings
warnings.filterwarnings("ignore")


DATA_DIR = "."

def knn(n):
    model = KNeighborsClassifier()  # default metric is Euclidean
    parameters = {'n_neighbors': [1,3,5]}
    return model, parameters, "kNN"

def logistic_regression(n):
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    parameters = {'C':loguniform(0.001, 1000), 'penalty': ["l2", "l1"], "solver": ["lbfgs", "sag"]}
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



def tune_hp(model_name, classifier, parameters, X_train, y_train, groups,
                   data_name, preprocessing):
    splitter = LeaveOneGroupOut()
    split = splitter.split(X_train, y_train, groups=groups)
    grid = RandomizedSearchCV(classifier, parameters, verbose=0, scoring="f1", cv=split)
    grid.fit(X_train, y_train)
    best_params = grid.best_params_
    str_params = {}
    for key, value in best_params.items():
        str_params[key] = str(value)
    filename = path.join(DATA_DIR, "evaluations/hyperparameters", f"{model_name}_{data_name}_{preprocessing}.json")
    with open(filename, 'wb') as f:
        json.dump(str_params, open(filename, "w"))

    
# ## Loading data representations

def integer_encoded(train_df):
    x_chen = pd.read_csv(path.join(DATA_DIR, "chen/integer_encoding/chen_integer_encoded.csv"), index_col=0)
    x_chen_train = x_chen.merge(train_df[["Antibody_ID", "Y", "cluster_merged"]], left_on="Ab_ID", right_on="Antibody_ID").drop("Antibody_ID", axis=1)
    return x_chen_train


def pybiomed(train_df):
    x_chen = pd.read_feather(path.join(DATA_DIR, "chen/pybiomed/X_data.ftr"))
    x_chen_train = x_chen.merge(train_df[["Antibody_ID", "Y", "cluster_merged"]], left_on="Ab_ID", right_on="Antibody_ID").drop("Antibody_ID", axis=1)
    return x_chen_train


def protparam(train_df):
    x_chen = pd.read_csv(path.join(DATA_DIR, "chen/protparam/protparam_features.csv"))
    x_chen.rename({"Unnamed: 0": "Ab_ID"}, axis=1, inplace=True)
    x_chen = x_chen.drop("name", axis=1)
    x_chen_train = x_chen.merge(train_df[["Antibody_ID", "Y", "cluster_merged"]], left_on="Ab_ID", right_on="Antibody_ID").drop("Antibody_ID", axis=1)
    
    return x_chen_train


def bert(train_df):
    x_chen = pd.read_feather(path.join(DATA_DIR, "chen/embeddings/bert/bert_chen_embeddings.ftr"))
    x_chen_train = x_chen.merge(train_df[["Antibody_ID", "Y", "cluster_merged"]], left_on="Ab_ID", right_on="Antibody_ID").drop("Antibody_ID", axis=1)
    return x_chen_train


def seqvec(train_df):
    x_chen = pd.read_feather(path.join(DATA_DIR, "chen/embeddings/seqvec/seqvec_chen_embeddings.ftr"))
    x_chen_train = x_chen.merge(train_df[["Antibody_ID", "Y", "cluster_merged"]], left_on="Ab_ID", right_on="Antibody_ID").drop("Antibody_ID", axis=1)
    return x_chen_train


def sapiens(train_df):
    x_chen = pd.read_csv(path.join(DATA_DIR, "chen/embeddings/sapiens/sapiens_chen_embeddings.csv"), index_col=0).drop("Y", axis=1)
    x_chen_train = x_chen.merge(train_df[["Antibody_ID", "Y", "cluster_merged"]], left_on="Ab_ID", right_on="Antibody_ID").drop("Antibody_ID", axis=1)
    return x_chen_train


def onehot(train_df):
    x_chen = pd.read_feather(path.join(DATA_DIR, "chen/onehot/chen_onehot_short.ftr"))
    x_chen_train = x_chen.merge(train_df[["Antibody_ID", "Y", "cluster_merged"]], left_on="Ab_ID", right_on="Antibody_ID").drop("Antibody_ID", axis=1)
    return x_chen_train


# ## Preprocessing

def no_prepro(train_df):
    return train_df.drop("Y", axis=1), train_df["Y"]


def scaling(train_df):
    scaler = StandardScaler()
    scaler.fit(train_df.drop(["Ab_ID", "Y", "cluster_merged"], axis=1))
    x_train_tr = scaler.transform(train_df.drop(["Ab_ID", "Y", "cluster_merged"], axis=1))
    x_train_df = pd.DataFrame(data=train_df,  index=train_df.index, columns=train_df.drop(["Ab_ID", "Y", "cluster_merged"], axis=1).columns)
    x_train_df["cluster_merged"] = train_df["cluster_merged"]
    x_train_df["Ab_ID"] = train_df["Ab_ID"]
    
    return x_train_df, train_df["Y"]


def oversampling(train_df):
    sampler = RandomOverSampler(random_state=42)
    x_train, y_train = sampler.fit_resample(train_df.drop("Y", axis=1), train_df["Y"])
    return x_train, y_train


def smote_os(train_df):
    sampler = SMOTE(random_state=42)
    x_train_tr, y_train = sampler.fit_resample(train_df.drop(["Ab_ID", "Y"], axis=1), train_df["Y"])
    x_train_tr["Ab_ID"] = ""
    return x_train_tr, y_train


def undersampling(train_df):
    sampler = RandomUnderSampler(random_state=42)
    x_train, y_train = sampler.fit_resample(train_df.drop("Y", axis=1), train_df["Y"]) 
    return x_train, y_train


data_loaders = [integer_encoded, pybiomed, protparam, bert, seqvec, sapiens, onehot]
model_creators = [knn, logistic_regression, random_forest, multilayer_perceptron, svm, gradient_boosting]
preprocessing = [no_prepro, scaling, oversampling, smote_os, undersampling]
  
# Input from console

CLI=argparse.ArgumentParser()
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

# # Load Data

chen_data = pd.read_csv(path.join(DATA_DIR, "chen/deduplicated/train_for_hpt.csv"), index_col=0)

def hpt_round(train_df):
    for prepro in preprocessing:
        prepro_name = prepro.__name__
        for data_rep in data_loaders:
            data_name = data_rep.__name__
            x_train = data_rep(train_df)
            x_train_tr, y_train_tr = prepro(x_train)
            n = len(x_train)
            for model_creator in model_creators:
                classifier, params, model_label = model_creator(n)
                tune_hp(
                    model_label, classifier, params, x_train_tr.drop(["Ab_ID", "cluster_merged"], axis=1), 
                    y_train_tr, x_train_tr["cluster_merged"], data_name, prepro_name
                )
                
hpt_round(chen_data)
            