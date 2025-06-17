# protein_embed_eval

This package allows users to analyze and visualize how different pretrained protein language models (e.g., ESM-2, ProtBERT, ProtGPT2) encode biological features like hydrophobicity, polarity, charge, and molecular weight.

## Features

- Load protein embeddings from different models
- Compute biologically relevant amino acid properties
- Cluster embeddings and analyze feature separation
- Evaluate with ANOVA, cosine similarity, and confidence intervals

## Example Usage

```python
from protein_embed_eval import embedding_loader, clustering, evaluation
yaml
Copy
Edit

