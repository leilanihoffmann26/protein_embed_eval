from protein_embed_eval import embedding_loader, clustering, feature_computation, visualization

# === Step 1: Generate and save embeddings ===
sequence = "ACDEFGHIKLMNPQRSTVWY"
embedding_path = "test_embeddings.npy"
embedding_loader.generate_embeddings(sequence)


# === Step 2: Run clustering ===
cluster_labels = clustering.cluster_embeddings(embedding_path, n_clusters=4, pca_dim=20)

# === Step 3: Compute feature relevance ===
feature_computation.analyze_features(sequence, cluster_labels, k=4)

# === Step 4: Visualize — 3D plot by feature ===
visualization.visualize_feature_embedding(
    embedding_path=embedding_path,
    sequence=sequence,
    feature_name="hydrophobicity",
    save_path="embedding_plot_hydro.html"
)

# Optional: Try another feature
visualization.visualize_feature_embedding(
    embedding_path=embedding_path,
    sequence=sequence,
    feature_name="polarity",
    save_path="embedding_plot_polarity.html"
)

