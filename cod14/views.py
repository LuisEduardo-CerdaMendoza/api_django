from django.shortcuts import render
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

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

def index(request):
    # Importar datos
    csv_file_path = os.path.join(os.path.dirname(__file__), '../archivojsn/TotalFeatures-ISCXFlowMeter.json')
    df = pd.read_json(csv_file_path)
    
    global last_f1_score

    train_set, val_set, test_set = train_val_test_split(df)
    X_train, y_train = remove_labels(train_set, 'calss')
    X_val, y_val = remove_labels(val_set, 'calss')
    X_test, y_test = remove_labels(test_set, 'calss')
    clf_rnd = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    clf_rnd.fit(X_train, y_train)
    y_pred = clf_rnd.predict(X_val)
    f1_score_val = f1_score(y_val, y_pred, average='weighted')
    last_f1_score = f1_score_val    

    return render(request, 'codigo14.html', {
        'f1_scores_val' : last_f1_score,
        'y_pred' : y_pred
    })
