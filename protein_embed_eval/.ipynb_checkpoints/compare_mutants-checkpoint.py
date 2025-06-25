# compare_mutants.py

import os
import numpy as np
import pandas as pd
import plotly.express as px
from pathlib import Path

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def load_embedding(path):
    return np.load(path)

def compare_cosine_similarity(wt_embedding, mut_embedding):
    return [cosine_similarity(w, m) for w, m in zip(wt_embedding, mut_embedding)]

def generate_similarity_diff_heatmap(wt_embedding, mut_embedding, sequence, output_path):
    sim_wt = np.inner(wt_embedding, wt_embedding)
    sim_mut = np.inner(mut_embedding, mut_embedding)
    delta_sim = sim_mut - sim_wt

    fig = px.imshow(
        delta_sim,
        labels=dict(x="Residue", y="Residue", color="Δ Cosine Similarity"),
        x=[f"{aa}{i+1}" for i, aa in enumerate(sequence)],
        y=[f"{aa}{i+1}" for i, aa in enumerate(sequence)],
        color_continuous_scale="RdBu",
        zmin=-1, zmax=1
    )

    for idx in [34, 51]:  # Glu35, Asp52 (0-based indexing)
        if idx < len(sequence):
            fig.add_shape(type="rect",
                          x0=idx - 0.5, x1=idx + 0.5,
                          y0=-0.5, y1=len(sequence) - 0.5,
                          line=dict(color="black", width=2))
            fig.add_shape(type="rect",
                          x0=-0.5, x1=len(sequence) - 0.5,
                          y0=idx - 0.5, y1=idx + 0.5,
                          line=dict(color="black", width=2))

    fig.update_layout(title="Δ Residue-Residue Cosine Similarity (Mutant - Wildtype)")
    fig.write_html(output_path)

def run_mutant_comparison(wt_dir, mutant_dir, sequence, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for model in ["ESM2", "ProtBERT", "ProtGPT2"]:
        wt_path = Path(wt_dir) / f"{model}_embedding.npy"
        mut_path = Path(mutant_dir) / f"{model}_embedding.npy"

        if wt_path.exists() and mut_path.exists():
            wt_emb = load_embedding(wt_path)
            mut_emb = load_embedding(mut_path)

            if wt_emb.shape != mut_emb.shape:
                print(f"Skipping {model}: Shape mismatch {wt_emb.shape} vs {mut_emb.shape}")
                continue

            similarities = compare_cosine_similarity(wt_emb, mut_emb)
            pd.DataFrame({
                "Residue": [f"{aa}{i+1}" for i, aa in enumerate(sequence)],
                "Cosine Similarity": similarities
            }).to_csv(Path(output_dir) / f"{model}_cosine_similarity.csv", index=False)

            generate_similarity_diff_heatmap(wt_emb, mut_emb, sequence, Path(output_dir) / f"{model}_similarity_diff_heatmap.html")
