import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def cluster_embeddings(
    embedding_path,
    n_clusters=4,
    pca_dim=None,
    return_reduced=False
):
    """
    Loads embeddings, optionally reduces dimensions, runs KMeans clustering.

    Args:
        embedding_path (str): Path to .npy file of shape (L, D)
        n_clusters (int): Number of clusters
        pca_dim (int or None): If set, reduces dimensions before clustering
        return_reduced (bool): If True, returns reduced embeddings too

    Returns:
        cluster_labels (np.array): Array of cluster labels for each residue
        (optional) reduced_embeddings
    """
    embeddings = np.load(embedding_path)
    original_shape = embeddings.shape

    # Standardize for PCA/KMeans
    X = StandardScaler().fit_transform(embeddings)

    # Dimensionality reduction
    if pca_dim is not None:
        pca = PCA(n_components=pca_dim)
        X_reduced = pca.fit_transform(X)
    else:
        X_reduced = X

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_reduced)

    if return_reduced:
        return labels, X_reduced
    return labels
