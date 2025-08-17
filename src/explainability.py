import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

import config
from model import TransformerAutoencoder

def analyze_anomaly_with_xai(model: TransformerAutoencoder, anomaly_sequence: np.ndarray):
    """
    Processes an anomalous sequence, visualizes the encoder's attention weights,
    and saves the visualization as a heatmap. 

    Args:
        model (TransformerAutoencoder): The trained autoencoder model.
        anomaly_sequence (np.ndarray): The single anomalous sequence to analyze.
                                       Shape: (sequence_length, num_features)
    """
    model.eval()
    
    # --- 1. Manually perform the initial steps of the forward pass ---
    seq_tensor = torch.tensor(anomaly_sequence, dtype=torch.float32).unsqueeze(0).to(config.DEVICE)
    
    # Project input to model dimension, same as in the model's forward pass
    src = model.input_projection(seq_tensor) * torch.sqrt(torch.tensor(model.model_dim, dtype=torch.float32))
    # src has shape (N, L, E) = (1, 100, 128) because of batch_first=True in the model

    # --- 2. Directly call the attention layer and request weights ---
    # Get the first encoder layer
    first_encoder_layer = model.transformer_encoder.layers[0]
    
    # The model's self-attention layer was initialized with `batch_first=True`,
    # so it expects the input tensor in the shape (N, L, E).
    # The previous permutation to (L, N, E) was incorrect and caused the shape error.
    # We now pass the `src` tensor directly without permutation.
    
    # Call the self-attention block directly with need_weights=True
    with torch.no_grad():
        # The input `src` is already in the correct (N, L, E) format.
        _, attn_weights = first_encoder_layer.self_attn(src, src, src, need_weights=True)
    
    if attn_weights is None:
        print("Error: Could not retrieve attention weights.")
        return

    # --- 3. Process and Visualize the Captured Weights ---
    # The weights now correctly have the shape: (N, L, S) = (1, 100, 100).
    
    # Squeeze the batch dimension to get the correct 2D (100, 100) heatmap.
    attention_map = attn_weights.squeeze(0).cpu().numpy()

    # Final check on the shape to prevent plotting errors.
    if attention_map.ndim != 2 or attention_map.shape[0] != attention_map.shape[1]:
        print(f"Error: Final attention map has incorrect shape: {attention_map.shape}. Expected a square 2D matrix.")
        return

    plt.figure(figsize=(12, 10))
    sns.heatmap(attention_map, cmap='viridis', xticklabels=False, yticklabels=False)
    plt.title("Attention Heatmap for Anomalous Sequence", fontsize=16)
    plt.xlabel("Key Positions (Time Steps)", fontsize=12)
    plt.ylabel("Query Positions (Time Steps)", fontsize=12)
    
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    save_path = os.path.join(config.RESULTS_DIR, 'attention_heatmap.png')
    plt.savefig(save_path)
    print(f"\nAttention heatmap for the detected anomaly saved to: {save_path}")
    # plt.show() # Uncomment to display the plot directly
