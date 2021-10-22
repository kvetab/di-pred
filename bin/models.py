from sklearn.linear_model import LogisticRegression
from sklearn.utils.fixes import loguniform    
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier


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