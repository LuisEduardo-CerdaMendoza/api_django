from django.shortcuts import render
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.gridspec as gridspec
from collections import Counter
from sklearn import metrics
import numpy as np
import os
import base64
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
# Create your views here.

def plot_data(X, y):
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], 'k.', markersize=2)
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], 'r.', markersize=2)

def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=30, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=50, linewidths=50,
                color=cross_color, zorder=11, alpha=1)

def plot_decision_boundaries(clusterer, X, y, resolution=1000, show_centroids=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                cmap="Pastel2")
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                linewidths=1, colors='k')
    plot_data(X, y)
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)
        
def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def index(request):
    # Importar datos
    csv_file_path = os.path.join(os.path.dirname(__file__), '../archivojsn/creditcard.csv')
    df = pd.read_csv(csv_file_path)
    
    global last_f1_score
    
    numerocaracteristicas = len(df.columns)
    longitudconjuntodatos = len(df)
    
    features = df.drop("Class", axis=1)

    plt.figure(figsize=(12,32))
    gs = gridspec.GridSpec(8, 4)
    gs.update(hspace=0.8)

    for i, f in enumerate(features):
        ax = plt.subplot(gs[i])
        sns.distplot(df[f][df["Class"] == 1])
        sns.distplot(df[f][df["Class"] == 0])
        ax.set_xlabel('')
        ax.set_title('feature: ' + str(f))
    plt.title('')
    plt.legend()
    # Guardar la imagen en el directorio estático de tu proyecto Django
    plt.savefig('graficas17.png')  # Guardar la imagen como PNG
    plt.close()

    with open("graficas17.png", "rb") as f:
        image_data = f.read()

    # Codificar la imagen en Base64
    graficas17 = base64.b64encode(image_data).decode()
    
    plt.figure(figsize=(12, 6))
    plt.scatter(df["V10"][df['Class'] == 0], df["V14"][df['Class'] == 0], c="g", marker=".")
    plt.scatter(df["V10"][df['Class'] == 1], df["V14"][df['Class'] == 1], c="r", marker=".")
    plt.xlabel("V10", fontsize=14)
    plt.ylabel("V14", fontsize=14)
    plt.title('')
    plt.legend()
    # Guardar la imagen en el directorio estático de tu proyecto Django
    plt.savefig('graficas172.png')  # Guardar la imagen como PNG
    plt.close()

    with open("graficas172.png", "rb") as f:
        image_data = f.read()

# Codificar la imagen en Base64
    graficas172 = base64.b64encode(image_data).decode()
    
    df = df.drop(["Time", "Amount"], axis=1)
    X = df[["V10", "V14"]].copy()
    
    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(X)
    
    plt.figure(figsize=(12, 6))
    plot_decision_boundaries(kmeans, X.values, df["Class"].values)
    plt.xlabel("V10", fontsize=14)
    plt.ylabel("V14", fontsize=14)
    
    plt.title('')
    plt.legend()
    # Guardar la imagen en el directorio estático de tu proyecto Django
    plt.savefig('graficas173.png')  # Guardar la imagen como PNG
    plt.close()

    with open("graficas173.png", "rb") as f:
        image_data = f.read()

    # Codificar la imagen en Base64
    graficas173 = base64.b64encode(image_data).decode()
    
    counter = Counter(clusters.tolist())
    bad_counter = Counter(clusters[df['Class'] == 1].tolist())

    for key in sorted(counter.keys()):
        print("Label {0} has {1} samples - {2} are malicious samples".format(
            key, counter[key], bad_counter[key]))
        
    X = df.drop("Class", axis=1)
    y = df["Class"].copy()
    
    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(X)
    
    counter = Counter(clusters.tolist())
    bad_counter = Counter(clusters[y == 1].tolist())

    for key in sorted(counter.keys()):
        print("Label {0} has {1} samples - {2} are malicious samples".format(
            key, counter[key], bad_counter[key]))
    
    clf_rnd = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    clf_rnd.fit(X, y)
    feature_importances = {name: score for name, score in zip(list(df), clf_rnd.feature_importances_)}
    feature_importances_sorted = pd.Series(feature_importances).sort_values(ascending=False)
    
    # Reducimos el conjunto de datos a las 7 características más importantes
    X_reduced = X[list(feature_importances_sorted.head(7).index)].copy()
    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(X_reduced)
    
    counter = Counter(clusters.tolist())
    bad_counter = Counter(clusters[y == 1].tolist())

    for key in sorted(counter.keys()):
        print("Label {0} has {1} samples - {2} are malicious samples".format(
            key, counter[key], bad_counter[key]))
        
    purityscore = purity_score(y, clusters)
    shiloutte = metrics.silhouette_score(X_reduced, clusters, sample_size=10000)
    calinski = metrics.calinski_harabasz_score(X_reduced, clusters)
    
    return render(request, 'codigo17.html', {
        'graficas17' : graficas17,
        'graficas172' : graficas172,
        'graficas173' : graficas173,
        'purityscore' : purityscore,
        'shiloutte' : shiloutte,
        'calinski' : calinski
    })