import os
import protein_embed_eval.feature_computation as feature_computation
from protein_embed_eval import embedding_loader, clustering, feature_computation, evaluation, visualization
import numpy as np

# Define the sequence
sequence = "ACDEFGHIKLMNPQRSTVWY"

# Output directory setup
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Step 1: Generate embeddings for all models
results = embedding_loader.benchmark_all_models(sequence)

label_dict = {}

# Step 2: Updated
for model_type in results.items():
    print(f"\n=== Processing {model_type} ===")
    embedding_path = f"examples/{model_type}_embedding.npy"

    # Cluster and visualize
    cluster_labels = clustering.cluster_embeddings(embedding_path, n_clusters=n_clusters, pca_dim=20)
    label_dict[model_type] = cluster_labels

    # Visualize distinguishing feature automatically
    best_feature = evaluation.find_best_feature(sequence, cluster_labels)
    output_html = f"examples/{model_type}_tsne_by_{best_feature}.html"
    visualization.visualize_embeddings_3d(embedding_path, sequence, feature=best_feature, output_html=output_html)

# Run ANOVA + CI analysis and save
anova_df, ci_df = feature_computation.run_feature_analysis(sequence, label_dict, k=n_clusters)
feature_computation.save_feature_stats(anova_df, ci_df, out_dir="examples")


# Step 2: For each model, run the pipeline
"""for model_type, emb in results.items():
    print(f"\n=== Processing {model_type} ===")

    model_dir = os.path.join(OUTPUT_DIR, model_type)
    os.makedirs(model_dir, exist_ok=True)

    emb_path = os.path.join(model_dir, f"{model_type}_embedding.npy")
    np.save(emb_path, emb)

    # Step 3: Clustering
    cluster_labels = clustering.cluster_embeddings(emb_path, n_clusters=4, pca_dim=20)

    # Step 4: Feature computation
    features_df = feature_computation.run_feature_analysis(sequence, label_dict, k=n_clusters)
    
    # Step 5: ANOVA
    anova_results = evaluation.compute_anova(features_df, cluster_labels)
    anova_results.to_csv(os.path.join(model_dir, f"{model_type}_anova.csv"), index=False)
    print(f"Saved ANOVA results to {model_dir}/{model_type}_anova.csv")

    # Step 6: Pick most significant feature
    best_feature_row = anova_results.sort_values("ANOVA p-value").iloc[0]
    best_feature = best_feature_row["Feature"]
    print(f"Most distinguishing feature for {model_type}: {best_feature}")

    # Step 7: Visualize embeddings colored by best feature
    output_html = os.path.join(model_dir, f"{model_type}_tsne_{best_feature}.html")
    output_png  = os.path.join(model_dir, f"{model_type}_tsne_{best_feature}.png")

    visualization.visualize_embeddings_3d(
        embedding_path=emb_path,
        sequence=sequence,
        feature=best_feature,
        output_html=output_html,
        output_png=output_png
    )"""
