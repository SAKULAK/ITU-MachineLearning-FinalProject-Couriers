import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from .data_manipulation import load_and_process_data
import matplotlib.pyplot as plt
import multiprocessing as mp
from joblib import Parallel, delayed

def pca_clustering_analysis(plot: tuple[bool, str] = (True, "Report/figs/"), verbose = True) -> plt.Figure:
    _, _, numeric_features, _, df = load_and_process_data("data/claims_train.csv", get_dummies=False, scaler=None, return_full_df=True)
    save_path = plot[1]+"pca_analysis.png"
    df = df.astype({col: float for col in numeric_features})
    
    if verbose:
        print("Performing PCA...")
    df.reset_index(inplace=True, drop=True)

    idx = df["Risk"].sort_values().index

    plt.figure(figsize=(18, 6))
    ContinousData = pd.DataFrame(StandardScaler().fit_transform(df[numeric_features]), columns=numeric_features)
    pca = PCA().fit(ContinousData)
    pcad = pca.transform(ContinousData)

    explained = np.cumsum(pca.explained_variance_ratio_)
    if verbose:
        print("Cumulative explained variance ratio by principal components:")
        print(explained)

    fig, axs = plt.subplots(1, 2)
    fig.set_figwidth(12)
    fig.set_figheight(6)

    if plot[0]:
        axs[0].scatter(pcad[idx, 0], pcad[idx, 1], c = df["Risk"].iloc[idx], alpha=0.25, cmap='cividis')
        axs[0].set_title("PCA visualisation of continous parameters")
        axs[0].set_xlabel("PC0")
        axs[0].set_ylabel("PC1")
        plt.colorbar(ax=axs[0], mappable=axs[0].collections[0], label='Risk')

    X_pca2 = pcad[:, :2]  # only keep first 2 principal components for clustering

    labels_list = list()

    
    for k in range(2, 13):
        if verbose:
            print(f"Clustering with k={k}...")
            verbosity = 3
        kmeans = KMeans(n_clusters=k, random_state=42, verbose=verbosity)
        labels_list.append((k, kmeans.fit_predict(X_pca2)))
        if verbose:
            print(f"Cluster centers for k={k}: {kmeans.cluster_centers_}")

    def silhouette_score_wrapper(X, labels):
        k, labels = labels
        print(f"Calculating silhouette score for {k=}...")
        score = silhouette_score(X, labels)
        return k, score

    silhouette_scores = Parallel(n_jobs=-1)(delayed(silhouette_score_wrapper)(X_pca2, labels) for labels in labels_list)

    best_k = max(silhouette_scores, key=lambda item: item[1])[0]
    if verbose:
        print(f"Best k found: {best_k}")

    kmeans = KMeans(n_clusters=best_k, random_state=42)
    clusters = kmeans.fit_predict(X_pca2)

    if plot[0]:
        axs[1].scatter(X_pca2[:, 0], X_pca2[:, 1], c=clusters, cmap="tab10", alpha=0.6)
        axs[1].set_xlabel("PC0")
        axs[1].set_ylabel("PC1")
        axs[1].set_title("KMeans Clusters for PCA")
        axs[1].grid(True)
        fig.tight_layout()
        fig.savefig(save_path)

    return fig