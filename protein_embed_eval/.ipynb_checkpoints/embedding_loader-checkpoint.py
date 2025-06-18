import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# Define a fixed sequence: 20 canonical amino acids
DEFAULT_SEQUENCE = "ACDEFGHIKLMNPQRSTVWY"

# Model configurations
MODEL_CONFIGS = {
    "ESM2": "facebook/esm2_t6_8M_UR50D",
    "ProtBERT": "Rostlab/prot_bert",
    "ProtGPT2": "nferruz/ProtGPT2"
}

def load_model_and_tokenizer(model_type):
    """Load the tokenizer and model based on model type."""
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model_name = MODEL_CONFIGS[model_type]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, use_safetensors=True) 
    return tokenizer, model


def embed_sequence(sequence, model_type="ESM2"):
    """Get per-residue embeddings from a sequence for a specified model."""
    tokenizer, model = load_model_and_tokenizer(model_type)
    model.eval()

    # Preprocess input
    if model_type == "ProtBERT":
        sequence = ' '.join(sequence)  # add spaces between residues
    elif model_type == "ProtGPT2":
        sequence = sequence  # raw input works fine
    # ESM2 does not need modifications

    inputs = tokenizer(sequence, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)
        hidden = outputs.last_hidden_state.squeeze(0)

        # Remove special tokens:
        if model_type == "ESM2":
            hidden = hidden[1:-1]  # remove [CLS], [EOS]
        elif model_type == "ProtBERT":
            hidden = hidden[1:-1]  # remove [CLS], [SEP]
        elif model_type == "ProtGPT2":
            # GPT2 is autoregressive, no special tokens trimming
            pass

    return hidden.numpy()  # shape: (seq_len, hidden_size)


import os

def benchmark_all_models(sequence=DEFAULT_SEQUENCE, save=True, out_dir="examples"):
    os.makedirs(out_dir, exist_ok=True)
    results = {}
    for model_type in MODEL_CONFIGS.keys():
        print(f"Embedding with {model_type}...")
        emb = embed_sequence(sequence, model_type)
        results[model_type] = emb

        if save:
            path = os.path.join(out_dir, f"{model_type}_embedding.npy")
            np.save(path, emb)
            print(f"Saved: {path}")

    return results
