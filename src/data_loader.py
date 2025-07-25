# src/data_loader.py

import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import config

def create_sequences(data, seq_length):
    """Converts a dataframe into sequences for the Transformer model."""
    sequences = []
    for i in range(len(data) - seq_length + 1):
        sequences.append(data.iloc[i:i + seq_length].values)
    return np.array(sequences)

def get_data_loaders(batch_size=config.BATCH_SIZE):
    """
    Loads, preprocesses, and prepares data loaders for training and testing.
    """
    # --- Training Data ---
    try:
        train_df = pd.read_csv(config.BENIGN_TRAFFIC_FILE)
        # Drop non-numeric columns or identifiers if they exist
        train_df = train_df.select_dtypes(include=np.number)
    except FileNotFoundError:
        print(f"Error: Training file not found at {config.BENIGN_TRAFFIC_FILE}. Please run a preprocessing script first.")
        return None, None, None

    # --- Feature Scaling ---
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df)
    train_scaled_df = pd.DataFrame(train_scaled, columns=train_df.columns)

    # --- Create Sequences for Training ---
    train_sequences = create_sequences(train_scaled_df, config.SEQUENCE_LENGTH)
    train_dataset = TensorDataset(torch.from_numpy(train_sequences).float())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # --- Testing Data (for evaluation) ---
    try:
        test_df = pd.read_csv(config.TEST_TRAFFIC_FILE)
        test_labels_str = test_df['Label']
        test_features = test_df.select_dtypes(include=np.number)
    except FileNotFoundError:
        print(f"Warning: Test file not found at {config.TEST_TRAFFIC_FILE}. Skipping test loader creation.")
        return train_loader, None, scaler

    # Use the *same* scaler from training data
    test_scaled = scaler.transform(test_features)
    test_scaled_df = pd.DataFrame(test_scaled, columns=test_features.columns)

    test_sequences = create_sequences(test_scaled_df, config.SEQUENCE_LENGTH)
    # Align labels with sequences. The first `seq_length - 1` labels are dropped.
    test_sequence_labels = test_labels_str.iloc[config.SEQUENCE_LENGTH - 1:].apply(lambda x: 0 if 'BENIGN' in str(x).upper() else 1).values

    test_dataset = TensorDataset(torch.from_numpy(test_sequences).float(), torch.from_numpy(test_sequence_labels).float())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print("DataLoaders created successfully.")
    return train_loader, test_loader, scaler

