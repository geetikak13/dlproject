# src/evaluate.py

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from model import TransformerAutoencoder
from data_loader import get_data_loaders
import config
import os
import json

def evaluate_model():
    """Evaluates the model on the test set and calculates classification metrics."""
    device = torch.device(config.DEVICE)
    
    # 1. Load Data and Model
    _, test_loader, _ = get_data_loaders()
    if test_loader is None:
        return

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

    # 2. Calculate Reconstruction Errors on Test Set
    criterion = nn.MSELoss(reduction='none')
    reconstruction_errors = []
    actual_labels = []

    print("--- Starting Evaluation ---")
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.to(device)
            reconstructed = model(sequences)
            loss = criterion(reconstructed, sequences)
            error_per_sequence = torch.mean(loss, dim=(1, 2))
            reconstruction_errors.extend(error_per_sequence.cpu().numpy())
            actual_labels.extend(labels.cpu().numpy())

    # 3. Determine Anomaly Threshold
    # This should ideally be done on a separate validation set of benign traffic.
    # Here, we'll use a percentile of the errors from the test set for demonstration.
    threshold = np.percentile(reconstruction_errors, 95)
    print(f"Anomaly Threshold (95th percentile): {threshold:.6f}")

    # 4. Classify and Evaluate
    predicted_labels = [1 if e > threshold else 0 for e in reconstruction_errors]
    
    precision, recall, f1, _ = precision_recall_fscore_support(actual_labels, predicted_labels, average='binary', zero_division=0)
    accuracy = accuracy_score(actual_labels, predicted_labels)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'anomaly_threshold': threshold
    }
    
    print("\n--- Evaluation Metrics ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("------------------------\n")

    # Save metrics to a file
    if not os.path.exists(config.RESULTS_PATH):
        os.makedirs(config.RESULTS_PATH)
    with open(os.path.join(config.RESULTS_PATH, 'performance_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {config.RESULTS_PATH}performance_metrics.json")

    # Plot confusion matrix
    cm = confusion_matrix(actual_labels, predicted_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'DDoS'], yticklabels=['Benign', 'DDoS'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    cm_path = os.path.join(config.RESULTS_PATH, 'confusion_matrix.png')
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")
    plt.show()


if __name__ == '__main__':
    evaluate_model()
