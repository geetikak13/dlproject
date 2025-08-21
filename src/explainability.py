import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

import config
from model import TransformerAutoencoder

def analyze_anomaly_with_xai(model: TransformerAutoencoder, anomaly_sequence: np.ndarray, attack_type: str):
    """
    Processes an anomalous sequence, visualizes the encoder's attention weights,
    and saves the visualization as a uniquely named heatmap for the given attack type.
    """
    model.eval()
    
    # --- 1. Manually perform the initial steps of the forward pass ---
    seq_tensor = torch.tensor(anomaly_sequence, dtype=torch.float32).unsqueeze(0).to(config.DEVICE)
    
    src = model.input_projection(seq_tensor) * torch.sqrt(torch.tensor(model.model_dim, dtype=torch.float32))

    # --- 2. Directly call the attention layer and request weights ---
    first_encoder_layer = model.transformer_encoder.layers[0]
    
    with torch.no_grad():
        _, attn_weights = first_encoder_layer.self_attn(src, src, src, need_weights=True)
    
    if attn_weights is None:
        print("Error: Could not retrieve attention weights.")
        return

    # --- 3. Process and Visualize the Captured Weights ---
    attention_map = attn_weights.squeeze(0).cpu().numpy()

    if attention_map.ndim != 2 or attention_map.shape[0] != attention_map.shape[1]:
        print(f"Error: Final attention map has incorrect shape: {attention_map.shape}. Expected a square 2D matrix.")
        return

    plt.figure(figsize=(12, 10))
    sns.heatmap(attention_map, cmap='viridis', xticklabels=10, yticklabels=10)
    plt.title(f"Attention Heatmap for Anomalous Sequence ({attack_type})", fontsize=16)
    plt.xlabel("Key Positions (Time Steps)", fontsize=12)
    plt.ylabel("Query Positions (Time Steps)", fontsize=12)
    
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    save_path = os.path.join(config.RESULTS_DIR, f'attention_heatmap_{attack_type}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"\nAttention heatmap for '{attack_type}' saved to: {save_path}")
