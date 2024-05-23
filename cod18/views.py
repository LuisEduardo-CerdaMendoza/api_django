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


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

# Create your views here.
def index(request):
    csv_file_path = os.path.join(os.path.dirname(__file__), '../archivojsn/creditcard.csv')
    df = pd.read_csv(csv_file_path)
    
    df["Class"].value_counts()
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
    plt.savefig('graficas18.png')  
    plt.close()

    with open("graficas18.png", "rb") as f:
        image_data = f.read()

    graficas18 = base64.b64encode(image_data).decode()
    
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x="V10", y="V14", hue="Class", palette=["g", "r"])
    plt.xlabel("V10", fontsize=14)
    plt.ylabel("V14", fontsize=14)
    
    plt.savefig('graficas182.png')
    plt.close()

    with open("graficas182.png", "rb") as f:
        image_data = f.read()

    graficas182 = base64.b64encode(image_data).decode()
    
    from sklearn.cluster import DBSCAN
    df = df.drop(["Time", "Amount"], axis=1)
    X = df[["V10", "V14"]].copy()
    y = df["Class"].copy()
    dbscan = DBSCAN(eps=0.15, min_samples=13)
    dbscan.fit(X)
    
    def plot_dbscan(dbscan, X, size):
        core_mask = np.zeros_like(dbscan.labels_, dtype=bool)
        core_mask[dbscan.core_sample_indices_] = True
        anomalies_mask = dbscan.labels_ == -1
        non_core_mask = ~(core_mask | anomalies_mask)

        cores = dbscan.components_
        anomalies = X[anomalies_mask]
        non_cores = X[non_core_mask]

        plt.figure(figsize=(12, 6))
        sns.scatterplot(x=cores[:, 0], y=cores[:, 1], hue=dbscan.labels_[core_mask], palette="Paired", size=size, legend=False)  
        plt.scatter(cores[:, 0], cores[:, 1], marker='*', s=20, c=dbscan.labels_[core_mask])
        plt.scatter(anomalies[:, 0], anomalies[:, 1], c="r", marker=".", s=100)
        plt.scatter(non_cores[:, 0], non_cores[:, 1], c=dbscan.labels_[non_core_mask], marker=".")
        plt.title("eps={:.2f}, min_samples={}".format(dbscan.eps, dbscan.min_samples), fontsize=14)
    plt.figure(figsize=(12, 6))
    plot_dbscan(dbscan, X.values, size=100)
    plt.xlabel("V10", fontsize=14)
    plt.ylabel("V14", fontsize=14)
    plt.savefig('graficas183.png')  
    plt.close()

    with open("graficas183.png", "rb") as f:
        image_data = f.read()

    graficas183 = base64.b64encode(image_data).decode()
    
    counter = Counter(dbscan.labels_.tolist())
    bad_counter = Counter(dbscan.labels_[y == 1].tolist())
    plot_dbscan(dbscan, X.values, size=100)
    
    for key in sorted(counter.keys()):
        print("Label {0} has {1} samples - {2} are malicious samples".format(
            key, counter[key], bad_counter[key]))
        
    X = df.drop("Class", axis=1)
    y = df["Class"].copy()
    
    from sklearn.ensemble import RandomForestClassifier
    clf_rnd = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    clf_rnd.fit(X, y)
    feature_importances = {name: score for name, score in zip(list(df), clf_rnd.feature_importances_)}
    feature_importances_sorted = pd.Series(feature_importances).sort_values(ascending=False)
    
    X_reduced = X[list(feature_importances_sorted.head(7).index)].copy()
    
    from sklearn.cluster import DBSCAN
    dbscan = DBSCAN(eps=0.70, min_samples=25)
    dbscan.fit(X_reduced)
    
    counter = Counter(dbscan.labels_.tolist())
    bad_counter = Counter(dbscan.labels_[y == 1].tolist())

    for key in sorted(counter.keys()):
        print("Label {0} has {1} samples - {2} are malicious samples".format(
            key, counter[key], bad_counter[key]))
        
    clusters = dbscan.labels_
    purity = purity_score(y, clusters)
    shiloutte = metrics.silhouette_score(X_reduced, clusters, sample_size=10000)
    calinski = metrics.calinski_harabasz_score(X_reduced, clusters)
    
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=1000, noise=0.05, random_state=42)
    
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette=["g", "r"])
    plt.savefig('graficas184.png')  
    plt.close()

    with open("graficas184.png", "rb") as f:
        image_data = f.read()

    graficas184 = base64.b64encode(image_data).decode()
    
    from sklearn.cluster import DBSCAN
    dbscan = DBSCAN(eps=0.1, min_samples=6)
    dbscan.fit(X)
    
    plt.figure(figsize=(12, 6))
    plot_dbscan(dbscan, X, size=100)
    plt.savefig('graficas185.png')  
    plt.close()

    with open("graficas185.png", "rb") as f:
        image_data = f.read()

    graficas185 = base64.b64encode(image_data).decode()
    
    counter = Counter(dbscan.labels_.tolist())
    bad_counter = Counter(dbscan.labels_[y == 1].tolist())

    for key in sorted(counter.keys()):
        print("Label {0} has {1} samples - {2} are malicious samples".format(
            key, counter[key], bad_counter[key]))
    
    return render(request, 'codigo18.html', {
    'graficas18' : graficas18,
    'graficas182' : graficas182,
    'graficas183' : graficas183,
    'graficas184' : graficas184,
    'graficas185' : graficas185,
    'purity' : purity,
    'shiloutte' : shiloutte,
    'calinski' : calinski
})