import os
import sys
import numpy as np
import plotly.graph_objects as go
from protein_embed_eval import embedding_loader, clustering, feature_computation, evaluation, visualization
from protein_embed_eval.residue_similarity import cosine_similarity_active_sites, cosine_similarity_matrix

# Handle optional command-line input
if len(sys.argv) > 1:
    sequence = sys.argv[1].strip().upper()
    is_default = False
else:
    # Hen egg white lysozyme
    sequence = "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSR"
    is_default = True

n_clusters = 4
EMBED_DIR = "examples"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Step 1: Generate embeddings
results = embedding_loader.benchmark_all_models(sequence)

# Step 2: Cluster, analyze, and visualize
label_dict = {}

for model_type, _ in results.items():
    print(f"\n=== Processing {model_type} ===")
    embedding_path = os.path.join(EMBED_DIR, f"{model_type}_embedding.npy")
    embedding = np.load(embedding_path)

    if embedding.shape[0] != len(sequence):
        print(f"Skipping {model_type}: embedding has {embedding.shape[0]} rows, expected {len(sequence)}.")
        continue

    # Cosine similarity of Glu35 and Asp52
    try:
        sim = cosine_similarity_active_sites(embedding)
        print(f"Cosine similarity between Glu35 and Asp52: {sim:.4f}")
        with open(os.path.join(OUTPUT_DIR, f"{model_type}_active_site_similarity.txt"), "w") as f:
            f.write(f"Glu35-Asp52 cosine similarity: {sim:.4f}\n")
    except IndexError as e:
        print(f"Skipping similarity calculation: {e}")

    # Clustering
    cluster_labels = clustering.cluster_embeddings(embedding_path, n_clusters=n_clusters, pca_dim=20)
    label_dict[model_type] = cluster_labels

    # Best feature + visualization
    best_feature = evaluation.find_best_feature(sequence, cluster_labels).lower()
    output_html = os.path.join(OUTPUT_DIR, f"{model_type}_tsne_by_{best_feature}.html")
    visualization.visualize_embeddings_3d(
        embedding_path=embedding_path,
        sequence=sequence,
        feature=best_feature,
        output_html=output_html
    )

    # === Residue similarity heatmap ===
    sim_matrix = cosine_similarity_matrix(embedding)
    residue_labels = [f"{aa}{i+1}" for i, aa in enumerate(sequence)]

    fig = go.Figure(
        data=go.Heatmap(
            z=sim_matrix,
            x=residue_labels,
            y=residue_labels,
            colorscale="RdBu",
            zmin=-1, zmax=1,
            colorbar=dict(title="Cosine Similarity"),
            hovertemplate="Residue 1: %{y}<br>Residue 2: %{x}<br>Similarity: %{z:.3f}<extra></extra>"
        )
    )

    fig.update_layout(
        title=f"{model_type} Residue Cosine Similarity",
        xaxis_title="Residue",
        yaxis_title="Residue",
        width=800,
        height=800
    )

    if is_default:
        # Highlight Glu35 and Asp52
        fig.add_trace(go.Scatter(
            x=["D52"], y=["E35"],
            mode="markers+text",
            marker=dict(color="yellow", size=12, line=dict(color="black", width=1)),
            text=["★"],
            textposition="middle center",
            hoverinfo="skip",
            showlegend=False
        ))

    heatmap_html = os.path.join(OUTPUT_DIR, f"{model_type}_residue_similarity_heatmap.html")
    fig.write_html(heatmap_html)
    print(f"Saved interactive heatmap to {heatmap_html}")

# Step 3: ANOVA + confidence intervals
anova_df, ci_df = feature_computation.run_feature_analysis(sequence, label_dict, k=n_clusters)
feature_computation.save_feature_stats(anova_df, ci_df, out_dir=OUTPUT_DIR)
