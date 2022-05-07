# Developability Index Prediction Project

Project in progress for master's thesis at University of Chemistry and Technology.

Problem and data come from [TDC](https://tdcommons.ai/single_pred_tasks/develop/).

The goal of the project is to train a model capable of predicting an antibody's developability index based on its protein sequence.

## Data

The `data` directory contains files necessary to run training. The `chen` directory contains the SAbDab / Chen et al. dataset along with all its data representations and train-test splits. The `tap` directory contains the TAP dataset and its representations.

Evaluation metrics and hyperparameters of trained models are stored in the `evaluations` directory. Actual trained models are not stored here. However, the best LSTM models are stored in the `models` directory. Fine-tuned ProteinBERT and Sapiens models are not present in this repository.

## Data processing

The first set of jupyter notebooks (in `notebooks/data_processing`) serve to download both datasets and create all respective data representations. Seven different data representations are created. These are then saved to the `data` directory.

Notebook 04 also produces the train-test splits based on clustering.

## Training

Six different ML models (k-NN, logistic regression, random forest, gradient boosting, SVM and multilayer perceptron) were trained for the classification task. Additionally, LSTM networks and two pre-trained transformer networks ([ProteinBERT](https://github.com/nadavbra/protein_bert) and [Sapiens](https://github.com/Merck/Sapiens)) were trained.

The first four notebooks and notebook 06 in the directory `notebooks/experiments` provide functionality for training ML models. In practice, these were trained using scripts in the `bin` directory, in particular `06_high_level_training.py`. `test_on_tap.py` was then used for testing these models.

LSTM models were trained using notebook 05 in `notebooks/experiments` and ProteinBERT was fine-tuned using notebook 09. 

All data and scripts for fine-tuning the Sapiens model are in `fairseq` directory. The notebook `sapiens_fine_tune.ipynb` also contains fine-tuning functionality, but serves mainly for evaluation.

## Evaluation

Data exploration and evaluation notebooks are in `notebooks/reports` along with various plots.