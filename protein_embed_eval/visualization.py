import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.io as pio

pio.renderers.default = 'notebook'  # Or 'browser' if outside notebooks

# --- Feature dictionaries ---
hydrophobicity = {
    'A': 1.8,  'C': 2.5,  'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5,  'K': -3.9, 'L': 3.8,
    'M': 1.9,  'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2,  'W': -0.9, 'Y': -1.3
}
charge = {
    'A': 0, 'C': 0, 'D': -1, 'E': -1, 'F': 0,
    'G': 0, 'H': 1, 'I': 0, 'K': 1, 'L': 0,
    'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'R': 1,
    'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0
}
molecular_weight = {
    'A': 89.1, 'C': 121.2, 'D': 133.1, 'E': 147.1, 'F': 165.2,
    'G': 75.1, 'H': 155.2, 'I': 131.2, 'K': 146.2, 'L': 131.2,
    'M': 149.2, 'N': 132.1, 'P': 115.1, 'Q': 146.2, 'R': 174.2,
    'S': 105.1, 'T': 119.1, 'V': 117.1, 'W': 204.2, 'Y': 181.2
}

feature_maps = {
    "hydrophobicity": hydrophobicity,
    "charge": charge,
    "molecular_weight": molecular_weight
}


def visualize_embeddings_3d(
    embedding_path,
    sequence,
    feature="hydrophobicity",
    perplexity=10,
    output_html=None
):
    # Load embeddings
    embeddings = np.load(embedding_path)

    # Get the feature mapping
    if feature not in feature_maps:
        raise ValueError(f"Feature '{feature}' not recognized. Choose from: {list(feature_maps.keys())}")
    feature_dict = feature_maps[feature]

    # Filter sequence + embeddings for valid residues
    clean_seq, clean_embeds, feature_vals = [], [], []
    for i, aa in enumerate(sequence):
        if aa in feature_dict and i < len(embeddings):
            clean_seq.append(aa)
            clean_embeds.append(embeddings[i])
            feature_vals.append(feature_dict[aa])

    clean_embeds = np.array(clean_embeds)

    # Reduce dimensions with t-SNE
    tsne = TSNE(n_components=3, random_state=42, perplexity=perplexity)
    tsne_3d = tsne.fit_transform(clean_embeds)

    # Plot
    fig = px.scatter_3d(
        x=tsne_3d[:, 0],
        y=tsne_3d[:, 1],
        z=tsne_3d[:, 2],
        color=feature_vals,
        color_continuous_scale="Viridis",
        labels={'color': feature.capitalize()},
        hover_name=[f"{aa}{i}" for i, aa in enumerate(clean_seq)],
        title=f"3D t-SNE of Protein Embeddings Colored by {feature.capitalize()}"
    )

    fig.show()

    if output_html:
        fig.write_html(output_html)
        print(f"Saved interactive plot to {output_html}")
