import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.special import softmax

import torch

from model import PANet


def PANet_scores(X):
    X = np.expand_dims(X, axis=0)

    xt = torch.tensor(X, dtype=torch.float32)
    device = torch.device("cpu")

    model = PANet().to(device)
    model.load_state_dict(torch.load('./PANet.pt', weights_only=False))
    model.eval()

    panet_scores = model(xt).detach().numpy().squeeze(0)

    return softmax(panet_scores)

def elbow_scores(X):
    inertias = []
    for k in range(1, 7):
        kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
        inertias.append(kmeans.inertia_)
    inertias = np.array(inertias)
    inverted = 1 - (inertias - inertias.min()) / (inertias.max() - inertias.min() + 1e-8)
    return softmax(inverted)

def silhouette_scores(X):
    sil_scores = [0.0]
    for k in range(2, 7):
        kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
        score = silhouette_score(X, kmeans.labels_)
        sil_scores.append(score)
    sil_scores = np.array(sil_scores)
    return softmax(sil_scores)

def compare_k_heatmap(X, y, num):
    ai_prob = PANet_scores(X)
    elbow_prob = elbow_scores(X)
    sil_prob = silhouette_scores(X)

    heatmap_data = np.vstack([ai_prob, sil_prob, elbow_prob])

    plt.figure(figsize=(10, 3))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="Greys", xticklabels=[f"K={k}" for k in range(1, 7)], yticklabels=["PANet", "Silhouette", "Elbow"])
    plt.title("Cluster Number Prediction Heatmap")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Method")
    plt.tight_layout()
    plt.savefig(f"./heatmaps/compare_heatmaps{num}.png", dpi=300)
    plt.close()



x = np.load('./datasets/X_test.npy')
y = np.load('./datasets/y_test.npy')
for i in range(100):
    compare_k_heatmap(x[i], np.argmax(y[i]), i+1)