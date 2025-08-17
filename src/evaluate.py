import torch
import torch.nn as nn
import numpy as np
import os
from sklearn.metrics import classification_report
from tqdm import tqdm

import config
from model import TransformerAutoencoder
from data_loader import load_and_preprocess_data
from explainability import analyze_anomaly_with_xai # Import the XAI function

def evaluate_model():
    """
    Evaluates the model by finding a reconstruction threshold, classifying a test set,
    and running the XAI module on the first detected anomaly.
    """
    # -- 1. Load Data to Determine Feature Count --
    print("Loading data to determine model dimensions...")
    X_benign = load_and_preprocess_data()
    
    # Dynamically determine the number of features from the loaded data.
    # This ensures the model architecture matches the saved checkpoint.
    num_features = X_benign.shape[2]
    print(f"Data loaded. Evaluating model with {num_features} features.")

    # -- 2. Load Model --
    print("Loading trained model...")
    model = TransformerAutoencoder(
        input_features=num_features, # Use the actual feature count
        model_dim=config.MODEL_DIM,
        num_heads=config.NUM_HEADS,
        num_encoder_layers=config.NUM_ENCODER_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        dropout=config.DROPOUT
    ).to(config.DEVICE)
    
    try:
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
    except FileNotFoundError:
        print(f"Model not found at {config.MODEL_SAVE_PATH}. Please run src/train.py first.")
        return
        
    model.eval()

    # -- 3. Determine Anomaly Threshold on Benign Data --
    print("\nDetermining anomaly threshold using a validation set of benign data...")
    # Use a subset of benign data as a validation set
    validation_data = torch.tensor(X_benign[:5000], dtype=torch.float32)

    reconstruction_errors = []
    with torch.no_grad():
        for i in tqdm(range(len(validation_data)), desc="Calculating validation errors"):
            seq = validation_data[i].unsqueeze(0).to(config.DEVICE)
            reconstruction = model(seq)
            error = nn.functional.mse_loss(reconstruction, seq).item()
            reconstruction_errors.append(error)
    
    # Set the threshold at the 99th percentile of errors on benign data
    anomaly_threshold = np.percentile(reconstruction_errors, 99)
    print(f"Anomaly Threshold (99th percentile): {anomaly_threshold:.6f}")

    # -- 4. Evaluate on a Simulated Test Set --
    print("\nEvaluating model on a simulated test set (benign + noisy data)...")
    # Simulate an attack set by adding noise to the benign validation data
    y_true = [0] * len(validation_data) # 0 for Benign
    
    noise = np.random.normal(0, 0.5, validation_data.shape)
    X_attack_simulated = validation_data.numpy() + noise
    y_true.extend([1] * len(X_attack_simulated)) # 1 for Anomaly
    
    X_test = np.concatenate([validation_data.numpy(), X_attack_simulated])
    
    y_pred = []
    first_anomaly_found = False
    with torch.no_grad():
        for i in tqdm(range(len(X_test)), desc="Classifying test data"):
            seq_np = X_test[i]
            seq = torch.tensor(seq_np, dtype=torch.float32).unsqueeze(0).to(config.DEVICE)
            reconstruction = model(seq)
            error = nn.functional.mse_loss(reconstruction, seq).item()
            
            is_anomaly = 1 if error > anomaly_threshold else 0
            y_pred.append(is_anomaly)
            
            # --- XAI INTEGRATION ---
            # If this is the first anomaly we've detected, run the explainability module
            if is_anomaly and not first_anomaly_found:
                print("\nFirst anomaly detected! Generating XAI attention heatmap...")
                analyze_anomaly_with_xai(model, seq_np)
                first_anomaly_found = True # Ensure we only do this once

    if not first_anomaly_found:
        print("\nNo anomalies were detected in the simulated test set.")

    # -- 5. Print Classification Report --
    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=['Benign', 'Anomaly']))

if __name__ == '__main__':
    evaluate_model()
