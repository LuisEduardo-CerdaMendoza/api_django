from django.shortcuts import render
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from matplotlib.colors import ListedColormap
from sklearn.ensemble import RandomForestClassifier
import base64
import io
import threading
from api.settings import BASE_DIR

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

def create_first_plot(df_reduced, y_df):
    plt.figure(figsize=(12, 6))
    plt.plot(df_reduced["c1"][y_df==0], df_reduced["c2"][y_df==0], "yo", label="normal")
    plt.plot(df_reduced["c1"][y_df==1], df_reduced["c2"][y_df==1], "bs", label="adware")
    plt.plot(df_reduced["c1"][y_df==2], df_reduced["c2"][y_df==2], "g^", label="malware")
    plt.xlabel("c1", fontsize=15)
    plt.ylabel("c2", fontsize=15, rotation=0)
    plt.title('Conjunto de datos con 2 características de entrada y 3 categorías')
    plt.legend()

    # Guardar la imagen como PNG
    plt.savefig('grafica1.png')

    # Cerrar la figura para liberar recursos
    plt.close()

def index(request):
    # Importar datos
    csv_file_path = os.path.join(os.path.dirname(__file__), '../archivojsn/TotalFeatures-ISCXFlowMeter.json')
    df = pd.read_json(csv_file_path)
    df['calss'].value_counts()
    
    global last_f1_score

    # Separamos las variables de entrada (X) de la etiqueta (y)
    X_df, y_df = remove_labels(df, 'calss')
    y_df = y_df.factorize()[0]    
    
    # Reducimos el conjunto de datos a 2 dimensiones utilizando el algoritmo PCA
    pca = PCA(n_components=2)
    df_reduced = pca.fit_transform(X_df)
    df_reduced = pd.DataFrame(df_reduced, columns=["c1", "c2"])

    # Crear la primera gráfica en un hilo separado
    threading.Thread(target=create_first_plot, args=(df_reduced, y_df)).start()

    with open("grafica1.png", "rb") as f:
        image_data = f.read()

    # Codificar la imagen en Base64
    grafica = base64.b64encode(image_data).decode()

    # Calculamos la proporción de varianza que se ha preservado del conjunto original
    varianza = pca.explained_variance_ratio_
    
    clf_tree_reduced = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf_tree_reduced.fit(df_reduced, y_df)
    
    def plot_decision_boundary(clf, X, y, plot_training=True, resolution=1000):
        mins = X.min(axis=0) - 1
        maxs = X.max(axis=0) + 1
        x1, x2 = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                            np.linspace(mins[1], maxs[1], resolution))
        X_new = np.c_[x1.ravel(), x2.ravel()]
        y_pred = clf.predict(X_new).reshape(x1.shape)
        custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
        plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
        if plot_training:
            plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", label="normal")
            plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", label="adware")
            plt.plot(X[:, 0][y==2], X[:, 1][y==2], "g^", label="malware")
            plt.axis([mins[0], maxs[0], mins[1], maxs[1]])               
        plt.xlabel(r"$x_1$", fontsize=18)
        plt.ylabel(r"$x_2$", fontsize=18, rotation=0)

    plt.figure(figsize=(12, 6))
    desicion = plot_decision_boundary(clf_tree_reduced, df_reduced.values, y_df)
    plt.title('Limite de Desicion')
    plt.legend()
    # Guardar la imagen en el directorio estático de tu proyecto Django
    plt.savefig('grafica2.png')  # Guardar la imagen como PNG
    plt.close()

    with open("grafica2.png", "rb") as f:
        image_data = f.read()

    # Codificar la imagen en Base64
    graficades = base64.b64encode(image_data).decode()
    
    # Reducimos el conjunto de datos manteniendo el 99,9% de varianza
    pca = PCA(n_components=0.999)
    df_reduced = pca.fit_transform(X_df)
    # Calculamos la proporción de varianza que se ha preservado del conjunto original
    pca.explained_variance_ratio_
    
    # Transformamos a un DataFrame de Pandas
    df_reduced = pd.DataFrame(df_reduced, columns=["c1", "c2", "c3", "c4", "c5", "c6"])
    df_reduced["Class"] = y_df
    df_reduced
    
    # Dividimos el conjunto de datos
    train_set, val_set, test_set = train_val_test_split(df_reduced)
    X_train, y_train = remove_labels(train_set, 'Class')
    X_val, y_val = remove_labels(val_set, 'Class')
    X_test, y_test = remove_labels(test_set, 'Class')
    
    
    clf_rnd = RandomForestClassifier(n_estimators=200, max_depth=30, random_state=42, n_jobs=-1)
    clf_rnd.fit(X_train, y_train)
    # Predecimos con el conjunto de datos de validación
    y_val_pred = clf_rnd.predict(X_val)
    
    # F1 score conjunto de datos de validación
    datosvalidacion =  f1_score(y_val_pred, y_val, average='weighted')
    # Predecimos con el conjunto de datos de pruebas
    y_test_pred = clf_rnd.predict(X_test)
    
    # F1 score conjunto de datos de pruebas
    last_f1_score = f1_score(y_test_pred, y_test, average='weighted')
    

    return render(request, 'codigo15.html', {
        'grafica' : grafica,
        'varianza' : varianza,
        'desicion' : desicion,
        'graficades' : graficades,
        'datosvalidacion' : datosvalidacion,
        'f1_scores_val' : last_f1_score
    })
