import torch
import torch.nn as nn
import numpy as np
import os
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
from glob import glob

import config
from model import TransformerAutoencoder
from data_loader import load_and_preprocess_data
from explainability import analyze_anomaly_with_xai

def get_predictions_in_batches(model, data, device, batch_size=256):
    """Helper function to get model predictions in efficient batches."""
    all_errors = []
    if not isinstance(data, np.ndarray):
        data = np.array(data)
        
    dataset = TensorDataset(torch.tensor(data, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size)
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Getting predictions in batches", leave=False):
            sequences = batch[0].to(device)
            reconstructions = model(sequences)
            errors = torch.mean((sequences - reconstructions) ** 2, dim=(1, 2))
            all_errors.extend(errors.cpu().numpy())
            
    return np.array(all_errors)

def evaluate_model():
    """
    Orchestrates the full evaluation process, using pre-cached data,
    efficient batch inference, and exporting the final results to a file.
    This version processes each attack file individually and saves a separate
    heatmap for each.
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
    validation_data = X_benign[:10000]
    validation_errors = get_predictions_in_batches(model, validation_data, config.DEVICE)
    anomaly_threshold = np.percentile(validation_errors, 99)
    print(f"Anomaly Threshold (99th percentile): {anomaly_threshold:.6f}")

    # -- Prepare results file --
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    report_path = os.path.join(config.RESULTS_DIR, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"Evaluation Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*50 + "\n")
        f.write(f"Anomaly Threshold (99th percentile): {anomaly_threshold:.6f}\n")

    # -- 3. Evaluation on Synthetic (Noisy) Data --
    print("\n--- Evaluation on Synthetic (Noisy) Data ---")
    test_benign_data_synth = X_benign[10000:20000]
    noise = np.random.normal(0, 0.5, test_benign_data_synth.shape)
    X_attack_synthetic = test_benign_data_synth + noise
    
    y_true_synth = [0] * len(test_benign_data_synth) + [1] * len(X_attack_synthetic)
    X_test_synth = np.concatenate([test_benign_data_synth, X_attack_synthetic])
    
    synth_errors = get_predictions_in_batches(model, X_test_synth, config.DEVICE)
    y_pred_synth = [1 if e > anomaly_threshold else 0 for e in synth_errors]
            
    acc_synth = accuracy_score(y_true_synth, y_pred_synth)
    report_synth = classification_report(y_true_synth, y_pred_synth, target_names=['Benign', 'Anomaly'])
    
    print(f"\nOverall Accuracy for Synthetic Data: {acc_synth * 100:.2f}%")
    print(report_synth)
    
    with open(report_path, 'a') as f:
        f.write("\n--- Evaluation on Synthetic (Noisy) Data ---\n")
        f.write(f"Overall Accuracy: {acc_synth * 100:.2f}%\n")
        f.write(report_synth)
        f.write("\n" + "="*50 + "\n")

    # -- 4. Evaluation on All Available Real Attack Data (One file at a time) --
    print("\n--- Evaluation on All Available Real Attack Data ---")
    
    attack_files = glob(os.path.join(config.PROCESSED_DATA_DIR, 'X_attack_*.npy'))
    if not attack_files:
        print("No processed attack files found. Please run `python src/data_loader.py` to generate them.")
        return

    all_y_true = []
    all_y_pred = []

    # Loop through each attack file individually
    for f_path in tqdm(attack_files, desc="Processing Attack Files"):
        attack_type = os.path.basename(f_path).replace('X_attack_', '').replace('.npy', '')
        tqdm.write(f"\nProcessing attack type: {attack_type}")

        X_attack_real = np.load(f_path)
        
        num_attack_samples = len(X_attack_real)
        benign_start_index = 20000 + (attack_files.index(f_path) * num_attack_samples)
        benign_end_index = benign_start_index + num_attack_samples
        
        if benign_end_index > len(X_benign):
            test_benign_data = X_benign[-num_attack_samples:]
        else:
            test_benign_data = X_benign[benign_start_index:benign_end_index]
        
        num_samples = min(len(test_benign_data), len(X_attack_real))
        test_benign_data = test_benign_data[:num_samples]
        X_attack_real = X_attack_real[:num_samples]

        y_true_current = [0] * num_samples + [1] * num_samples
        X_test_current = np.concatenate([test_benign_data, X_attack_real])

        current_errors = get_predictions_in_batches(model, X_test_current, config.DEVICE)
        y_pred_current = [1 if e > anomaly_threshold else 0 for e in current_errors]

        # Trigger XAI for the first anomaly found IN THIS FILE
        first_anomaly_index_in_file = next((i for i, pred in enumerate(y_pred_current) if pred == 1 and y_true_current[i] == 1), None)
        if first_anomaly_index_in_file is not None:
            analyze_anomaly_with_xai(model, X_test_current[first_anomaly_index_in_file], attack_type)

        all_y_true.extend(y_true_current)
        all_y_pred.extend(y_pred_current)

    # -- 5. Final Aggregated Report --
    print("\n--- Aggregated Report for All Real Attack Data ---")
    
    acc_real = accuracy_score(all_y_true, all_y_pred)
    report_real = classification_report(all_y_true, all_y_pred, target_names=['Benign', 'Anomaly'])
    
    print(f"Tested against {len(attack_files)} attack types.")
    print(f"Overall Aggregated Accuracy: {acc_real * 100:.2f}%")
    print(report_real)

    with open(report_path, 'a') as f:
        f.write("\n--- Aggregated Report for All Real Attack Data ---\n")
        f.write(f"Tested against {len(attack_files)} attack types.\n")
        f.write(f"Overall Aggregated Accuracy: {acc_real * 100:.2f}%\n")
        f.write(report_real)

    print(f"\nEvaluation complete. Report saved to {report_path}")

if __name__ == '__main__':
    evaluate_model()
