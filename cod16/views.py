from django.shortcuts import render
import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint

def train_val_test_split(df, rstate=42, shuffle=True, stratify=None):
    strat = df[stratify] if stratify else None
    train_set, test_set = train_test_split(
        df, test_size=0.4, random_state=rstate, shuffle=shuffle, stratify=strat)
    strat = test_set[stratify] if stratify else None
    val_set, test_set = train_test_split(
        test_set, test_size=0.5, random_state=rstate, shuffle=shuffle, stratify=strat)
    return (train_set, val_set, test_set)

def remove_labels(df, label_name):
    X = df.drop(label_name, axis=1)
    y = df[label_name].copy()
    return (X, y)

def search_hyperparameters(clf, param_grid, X_train, y_train):
    grid_search = GridSearchCV(clf, param_grid, cv=5,
                            scoring='f1_weighted', return_train_score=True)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def search_random_hyperparameters(clf, param_distribs, X_train, y_train):
    rnd_search = RandomizedSearchCV(clf, param_distributions=param_distribs,
                                    n_iter=5, cv=2, scoring='f1_weighted')
    rnd_search.fit(X_train, y_train)
    return rnd_search.best_estimator_

def calculate_f1_score(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted')

def index(request):
    # Importar datos
    csv_file_path = os.path.join(os.path.dirname(__file__), '../archivojsn/TotalFeatures-ISCXFlowMeter.json')
    df = pd.read_json(csv_file_path).sample(n=3000, random_state=42)
    
    longitudconjdatos = len(df)
    numcaracter = len(df.columns)
    df['calss'].value_counts()
    
    train_set, val_set, test_set = train_val_test_split(df)
    X_train, y_train = remove_labels(train_set, 'calss')
    X_val, y_val = remove_labels(val_set, 'calss')
    X_test, y_test = remove_labels(test_set, 'calss')
    
    # Modelo entrenado con el conjunto de datos sin escalar
    clf_rnd = RandomForestClassifier(n_estimators=3000, random_state=42, n_jobs=-1)
    clf_rnd.fit(X_train, y_train)
    
    # Predecimos con el conjunto de datos de validación
    y_pred = clf_rnd.predict(X_val)
    conjuntodatos = calculate_f1_score(y_pred, y_val)
    
    # Búsqueda de hiperparámetros
    param_grid = [
        {'n_estimators': [100, 500, 1000], 'max_leaf_nodes': [16, 24, 36]},
        {'bootstrap': [False], 'n_estimators': [100, 500], 'max_features': [2, 3, 4]},
    ]

    clf_rnd = search_hyperparameters(RandomForestClassifier(n_jobs=-1, random_state=42), param_grid, X_train, y_train)
    
    param_distribs = {
            'n_estimators': randint(low=1, high=200),
            'max_depth': randint(low=8, high=50),
        }

    clf_rnd = search_random_hyperparameters(RandomForestClassifier(n_jobs=-1), param_distribs, X_train, y_train)
    
    # Cálculo del F1-score para el conjunto de entrenamiento
    y_train_pred = clf_rnd.predict(X_train)
    prediccionconjdatos = calculate_f1_score(y_train_pred, y_train)
    
    # F1 score conjunto de datos de pruebas
    last_f1_score = calculate_f1_score(clf_rnd.predict(X_val), y_val)

    return render(request, 'codigo16.html', {
        'longitudconjdatos': longitudconjdatos,
        'numcaracter': numcaracter,
        'conjuntodatos': conjuntodatos,
        'prediccionconjdatos': prediccionconjdatos,
        'f1_scores_val': last_f1_score
    })