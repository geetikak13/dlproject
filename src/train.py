# src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from model import TransformerAutoencoder
from data_loader import get_data_loaders
import config
import os

def train_model():
    """Main training loop for the Transformer Autoencoder."""
    # 1. Setup device, data, and model
    device = torch.device(config.DEVICE)
    train_loader, _, _ = get_data_loaders()

    if train_loader is None:
        return

    model = TransformerAutoencoder(
        input_dim=config.INPUT_DIM,
        model_dim=config.MODEL_DIM,
        nhead=config.NUM_HEADS,
        num_encoder_layers=config.NUM_ENCODER_LAYERS,
        num_decoder_layers=config.NUM_DECODER_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        dropout=config.DROPOUT
    ).to(device)

    # 2. Define Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # 3. Training Loop
    print("--- Starting Model Training ---")
    model.train()
    for epoch in range(config.EPOCHS):
        total_loss = 0
        for i, (batch,) in enumerate(train_loader):
            sequences = batch.to(device)

            # Forward pass
            optimizer.zero_grad()
            reconstructed = model(sequences)
            loss = criterion(reconstructed, sequences)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1:02d}/{config.EPOCHS}], Loss: {avg_loss:.6f}')

    # 4. Save the trained model
    if not os.path.exists(config.MODEL_PATH):
        os.makedirs(config.MODEL_PATH)
    save_path = os.path.join(config.MODEL_PATH, config.MODEL_NAME)
    torch.save(model.state_dict(), save_path)
    print(f"--- Model Training Finished ---")
    print(f"Model saved to {save_path}")

if __name__ == '__main__':
    train_model()
