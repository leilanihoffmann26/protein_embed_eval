import os
import numpy as np
from protein_embed_eval import embedding_loader, clustering, feature_computation, evaluation, visualization

# Define the sequence and parameters
sequence = "ACDEFGHIKLMNPQRSTVWY"
n_clusters = 4

EMBED_DIR = "examples"

# Set output directory
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Step 1: Generate embeddings
results = embedding_loader.benchmark_all_models(sequence)

# Step 2: Cluster and visualize embeddings
label_dict = {}

for model_type, _ in results.items():
    print(f"\n=== Processing {model_type} ===")
    embedding_path = os.path.join(EMBED_DIR, f"{model_type}_embedding.npy")

    # Cluster
    cluster_labels = clustering.cluster_embeddings(embedding_path, n_clusters=n_clusters, pca_dim=20)
    label_dict[model_type] = cluster_labels

    # Automatically pick best feature and visualize
    best_feature = evaluation.find_best_feature(sequence, cluster_labels)
    output_html = os.path.join(OUTPUT_DIR, f"{model_type}_tsne_by_{best_feature}.html")
    best_feature = best_feature.lower()
    visualization.visualize_embeddings_3d(
        embedding_path=embedding_path,
        sequence=sequence,
        feature=best_feature.lower(),
        output_html=output_html
    )

# Step 3: Feature analysis
anova_df, ci_df = feature_computation.run_feature_analysis(sequence, label_dict, k=n_clusters)
feature_computation.save_feature_stats(anova_df, ci_df, out_dir=OUTPUT_DIR)
