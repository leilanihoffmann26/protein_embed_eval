# feature_computation.py
import os
import numpy as np
import pandas as pd
from scipy.stats import f_oneway, t

# --- Feature Dictionaries ---
hydrophobicity = {
    'A': 1.8,  'C': 2.5,  'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5,  'K': -3.9, 'L': 3.8,
    'M': 1.9,  'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2,  'W': -0.9, 'Y': -1.3
}
polarity = {
    'A': 0, 'C': 0, 'D': 1, 'E': 1, 'F': 0, 'G': 0, 'H': 1, 'I': 0,
    'K': 1, 'L': 0, 'M': 0, 'N': 1, 'P': 0, 'Q': 1, 'R': 1,
    'S': 1, 'T': 1, 'V': 0, 'W': 0, 'Y': 1
}
charge = {
    'A': 0, 'C': 0, 'D': -1, 'E': -1, 'F': 0, 'G': 0, 'H': 1, 'I': 0,
    'K': 1, 'L': 0, 'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'R': 1,
    'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0
}
molecular_weight = {
    'A': 89.1, 'C': 121.2, 'D': 133.1, 'E': 147.1, 'F': 165.2,
    'G': 75.1, 'H': 155.2, 'I': 131.2, 'K': 146.2, 'L': 131.2,
    'M': 149.2, 'N': 132.1, 'P': 115.1, 'Q': 146.2, 'R': 174.2,
    'S': 105.1, 'T': 119.1, 'V': 117.1, 'W': 204.2, 'Y': 181.2
}

def extract_features(sequence):
    """Convert sequence into numeric biochemical property arrays."""
    return {
        'Hydrophobicity': np.array([hydrophobicity[aa] for aa in sequence]),
        'Polarity': np.array([polarity[aa] for aa in sequence]),
        'Charge': np.array([charge[aa] for aa in sequence]),
        'Molecular Weight': np.array([molecular_weight[aa] for aa in sequence])
    }

def anova_p(values, labels, k):
    """Return ANOVA p-value across k clusters."""
    groups = [values[labels == i] for i in range(k)]
    return f_oneway(*groups).pvalue

def compute_ci(values, labels, k):
    """Return (mean, 95% CI) for each cluster."""
    results = []
    for i in range(k):
        group = values[labels == i]
        mean = np.mean(group)
        std = np.std(group, ddof=1)
        n = len(group)
        if n > 1:
            ci = t.ppf(0.975, df=n-1) * std / np.sqrt(n)
        else:
            ci = np.nan
        results.append((round(mean, 3), round(ci, 3)))
    return results

def run_feature_analysis(sequence, label_dict, k):
    """
    Runs ANOVA and 95% CI feature analysis.
    
    Args:
        sequence: str - protein sequence
        label_dict: dict - model_name -> cluster_labels (array)
        k: int - number of clusters
    
    Returns:
        anova_df: pd.DataFrame
        ci_df: pd.DataFrame
    """
    features = extract_features(sequence)

    # ANOVA
    anova_results = {'Model': [], 'Feature': [], 'ANOVA p-value': []}
    for model_name, labels in label_dict.items():
        for fname, values in features.items():
            pval = anova_p(values, labels, k)
            anova_results['Model'].append(model_name)
            anova_results['Feature'].append(fname)
            anova_results['ANOVA p-value'].append(round(pval, 4))
    anova_df = pd.DataFrame(anova_results)

    # Confidence Intervals
    ci_data = {'Model': [], 'Feature': [], 'Cluster': [], 'Mean ± 95% CI': []}
    for model_name, labels in label_dict.items():
        for fname, values in features.items():
            cluster_cis = compute_ci(values, labels, k)
            for i, (mean, ci) in enumerate(cluster_cis):
                ci_data['Model'].append(model_name)
                ci_data['Feature'].append(fname)
                ci_data['Cluster'].append(f'Cluster {i}')
                ci_data['Mean ± 95% CI'].append(f'{mean} ± {ci}')
    ci_df = pd.DataFrame(ci_data)

    return anova_df, ci_df

def save_feature_stats(anova_df, ci_df, out_dir="examples"):
    os.makedirs(out_dir, exist_ok=True)
    anova_df.to_csv(os.path.join(out_dir, "anova_results.csv"), index=False)
    ci_df.to_csv(os.path.join(out_dir, "confidence_intervals.csv"), index=False)
    print("Saved ANOVA and confidence intervals to examples/")

