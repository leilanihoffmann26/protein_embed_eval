# evaluation.py

from protein_embed_eval import feature_computation

def find_best_feature(sequence, cluster_labels, k=4):
    """
    Returns the feature with the smallest ANOVA p-value.
    
    Args:
        sequence (str): Protein sequence.
        cluster_labels (np.array): Cluster labels for residues.
        k (int): Number of clusters.
    
    Returns:
        str: Name of the most distinguishing feature.
    """
    label_dict = {"temp_model": cluster_labels}
    anova_df, _ = feature_computation.run_feature_analysis(sequence, label_dict, k=k)
    best_feature_row = anova_df.loc[anova_df["ANOVA p-value"].idxmin()]
    return best_feature_row["Feature"]
