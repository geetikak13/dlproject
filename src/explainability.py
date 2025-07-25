# src/explainability.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model import TransformerAutoencoder
from data_loader import get_data_loaders
import config
import os

def visualize_attention():
    """
    Visualizes the attention weights for the first anomalous sequence found in the test set.
    """
    print("--- Running Explainability Module ---")
    device = torch.device(config.DEVICE)
    
    # 1. Load a single sequence for analysis
    _, test_loader, _ = get_data_loaders(batch_size=1) # Use batch size 1 to easily pick one sequence
    if test_loader is None:
        return

    # Find the first anomalous sequence
    sample_sequence, sample_label = None, None
    sequence_idx = -1
    for i, (seq, lbl) in enumerate(test_loader):
        if lbl.item() == 1: # 1 indicates DDoS/anomaly
            sample_sequence = seq
            sample_label = lbl
            sequence_idx = i
            break
            
    if sample_sequence is None:
        print("No anomalous sequence found in the test set.")
        return
        
    sample_sequence = sample_sequence.to(device)

    # 2. Load the trained model
    model = TransformerAutoencoder(
        input_dim=config.INPUT_DIM,
        model_dim=config.MODEL_DIM,
        nhead=config.NUM_HEADS,
        num_encoder_layers=config.NUM_ENCODER_LAYERS,
        num_decoder_layers=config.NUM_DECODER_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD
    ).to(device)
    
    model_path = os.path.join(config.MODEL_PATH, config.MODEL_NAME)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}. Please run train.py first.")
        return
        
    model.eval()

    # 3. Get attention weights from the model
    print(f"Fetching attention weights for anomalous sequence index {sequence_idx}...")
    _, attention_weights = model(sample_sequence, return_attention=True)
    
    if not attention_weights:
        print("\n[Error] Could not retrieve attention weights. Ensure hooks are set up correctly in the model.")
        return
    
    # 4. Plot the attention maps
    if not os.path.exists(config.ATTENTION_MAP_DIR):
        os.makedirs(config.ATTENTION_MAP_DIR)

    for i, att_map in enumerate(attention_weights):
        # att_map shape is (batch_size, num_heads, seq_len, seq_len)
        # We average across all heads for visualization.
        att_map = att_map.squeeze(0).mean(dim=0).cpu().detach().numpy()

        plt.figure(figsize=(12, 10))
        sns.heatmap(att_map, cmap='viridis')
        plt.title(f'Attention Map - Encoder Layer {i+1} for Sequence {sequence_idx}')
        plt.xlabel('Key (Attended To)')
        plt.ylabel('Query (Attending From)')
        
        save_path = os.path.join(config.ATTENTION_MAP_DIR, f'attention_map_layer_{i+1}_seq_{sequence_idx}.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Saved attention map for layer {i+1} to {save_path}")

if __name__ == '__main__':
    visualize_attention()

