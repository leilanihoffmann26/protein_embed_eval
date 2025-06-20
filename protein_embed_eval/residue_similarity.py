import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def cosine_similarity_active_sites(embedding, glu_index=34, asp_index=51):
    """
    Compute cosine similarity between Glu35 and Asp52 (0-based indices).
    """
    if embedding.ndim != 2:
        raise ValueError("Embedding must be a 2D array (residues × features).")

    if glu_index >= embedding.shape[0] or asp_index >= embedding.shape[0]:
        raise IndexError("One of the active site indices is out of bounds for the sequence.")

    vec_glu = embedding[glu_index].reshape(1, -1)
    vec_asp = embedding[asp_index].reshape(1, -1)

    similarity = cosine_similarity(vec_glu, vec_asp)[0][0]
    return similarity

def cosine_similarity_matrix(embedding):
    """
    Compute pairwise cosine similarity between all residue embeddings.

    Args:
        embedding (np.ndarray): Shape (L, D), where L is the number of residues.

    Returns:
        np.ndarray: (L, L) matrix of cosine similarities between residues.
    """
    return cosine_similarity(embedding)
