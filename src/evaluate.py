import torch
import torch.nn as nn
import numpy as np
import os
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

import config
from model import TransformerAutoencoder
from data_loader import load_and_preprocess_data, load_and_preprocess_attack_data
from explainability import analyze_anomaly_with_xai

def run_evaluation(model, anomaly_threshold, benign_data, attack_data, test_name, trigger_xai=False):
    """
    Helper function to run an evaluation on a given set of benign and attack data.
    """
    print(f"\n--- Evaluation on {test_name} ---")
    
    y_true = [0] * len(benign_data)  # 0 for Benign
    y_true.extend([1] * len(attack_data)) # 1 for Anomaly
    
    X_test = np.concatenate([benign_data, attack_data])
    
    y_pred = []
    first_anomaly_found = False
    with torch.no_grad():
        for i in tqdm(range(len(X_test)), desc=f"Classifying {test_name}"):
            seq_np = X_test[i]
            seq = torch.tensor(seq_np, dtype=torch.float32).unsqueeze(0).to(config.DEVICE)
            reconstruction = model(seq)
            error = nn.functional.mse_loss(reconstruction, seq).item()
            
            is_anomaly = 1 if error > anomaly_threshold else 0
            y_pred.append(is_anomaly)
            
            # Trigger XAI module on the first detected anomaly if requested
            if is_anomaly and not first_anomaly_found and trigger_xai:
                print(f"\nFirst anomaly in '{test_name}' detected! Generating XAI attention heatmap...")
                analyze_anomaly_with_xai(model, seq_np)
                first_anomaly_found = True

    acc = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy for {test_name}: {acc * 100:.2f}%")
    print(f"\nClassification Report for {test_name}:")
    print(classification_report(y_true, y_pred, target_names=['Benign', 'Anomaly']))
    
    if trigger_xai and not first_anomaly_found:
        print(f"\nNo anomalies were detected in the {test_name} set.")

def evaluate_model():
    """
    Orchestrates the full evaluation process, comparing performance on synthetic
    and real attack data.
    """
    # -- 1. Load Data and Model --
    print("Loading data and model...")
    X_benign = load_and_preprocess_data()
    num_features = X_benign.shape[2]
    
    model = TransformerAutoencoder(
        input_features=num_features, model_dim=config.MODEL_DIM, num_heads=config.NUM_HEADS,
        num_encoder_layers=config.NUM_ENCODER_LAYERS, dim_feedforward=config.DIM_FEEDFORWARD,
        dropout=config.DROPOUT
    ).to(config.DEVICE)
    
    try:
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
    except FileNotFoundError:
        print(f"Model not found at {config.MODEL_SAVE_PATH}. Please run src/train.py first.")
        return
    model.eval()

    # -- 2. Determine Anomaly Threshold --
    print("\nDetermining anomaly threshold...")
    validation_data = X_benign[:5000]
    reconstruction_errors = []
    with torch.no_grad():
        for i in tqdm(range(len(validation_data)), desc="Calculating validation errors"):
            seq = torch.tensor(validation_data[i], dtype=torch.float32).unsqueeze(0).to(config.DEVICE)
            reconstruction = model(seq)
            error = nn.functional.mse_loss(reconstruction, seq).item()
            reconstruction_errors.append(error)
    
    anomaly_threshold = np.percentile(reconstruction_errors, 99)
    print(f"Anomaly Threshold (99th percentile): {anomaly_threshold:.6f}")

    # -- 3. Run Evaluation on Synthetic (Noisy) Data --
    noise = np.random.normal(0, 0.5, validation_data.shape)
    X_attack_synthetic = validation_data + noise
    run_evaluation(model, anomaly_threshold, validation_data, X_attack_synthetic, "Synthetic (Noisy) Data")

    # -- 4. Run Evaluation on Real Attack Data --
    try:
        attack_type_to_test = 'TFTP'
        X_attack_real = load_and_preprocess_attack_data(attack_type_to_test)
        # Use the same number of samples for a balanced test
        X_attack_real_sample = X_attack_real[:len(validation_data)]
        
        # This is the main test, so we trigger the XAI here.
        run_evaluation(model, anomaly_threshold, validation_data, X_attack_real_sample, f"Real Attack Data ({attack_type_to_test})", trigger_xai=True)
    except (FileNotFoundError, ValueError) as e:
        print(f"\nCould not perform evaluation on real attack data: {e}")

if __name__ == '__main__':
    evaluate_model()
