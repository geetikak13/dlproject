import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from tqdm import tqdm

import config
from model import TransformerAutoencoder
from data_loader import load_and_preprocess_data

def train_model():
    """
    Orchestrates the model training process.
    """
    # -- 1. Load Data --
    X_train = load_and_preprocess_data()

    # Dynamically determine the number of features from the loaded data.
    # This prevents the shape mismatch error if the preprocessed data has a
    # different number of features than specified in the config file.
    num_features = X_train.shape[2]
    print(f"Data loaded. Training model with {num_features} features.")
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    train_dataset = TensorDataset(X_train_tensor, X_train_tensor) # Input and target are the same
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    # -- 2. Initialize Model --
    model = TransformerAutoencoder(
        input_features=num_features, # Use the actual feature count
        model_dim=config.MODEL_DIM,
        num_heads=config.NUM_HEADS,
        num_encoder_layers=config.NUM_ENCODER_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        dropout=config.DROPOUT
    ).to(config.DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.MSELoss() # Using Mean Squared Error for reconstruction loss

    print("Starting model training...")
    # -- 3. Training Loop --
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        for batch_inputs, batch_targets in progress_bar:
            batch_inputs = batch_inputs.to(config.DEVICE)
            batch_targets = batch_targets.to(config.DEVICE)
            
            # Forward pass
            reconstructions = model(batch_inputs)
            loss = criterion(reconstructions, batch_targets)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}], Average Loss: {avg_loss:.6f}")

    # -- 4. Save Model --
    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    print(f"Model saved to {config.MODEL_SAVE_PATH}")

if __name__ == '__main__':
    train_model()
