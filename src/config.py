# src/config.py

import torch

# -- Data Configuration --
# NOTE: You will need to create these processed files yourself from the original CIC-DDoS2019 dataset.
# This typically involves cleaning, feature selection, and splitting the data.
DATA_DIR = '../data/processed/'
BENIGN_TRAFFIC_FILE = DATA_DIR + 'benign_traffic_processed.csv'
TEST_TRAFFIC_FILE = DATA_DIR + 'test_traffic_processed.csv' # This file should contain both benign and DDoS samples

# -- Model Saving Configuration --
MODEL_PATH = '../saved_models/'
MODEL_NAME = 'transformer_autoencoder.pth'
RESULTS_PATH = '../results/'
ATTENTION_MAP_DIR = RESULTS_PATH + 'attention_maps/'

# -- Model Hyperparameters --
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
SEQUENCE_LENGTH = 50 # The number of time steps in each sequence
INPUT_DIM = 78 # Example: Number of features in the CIC-DDoS2019 dataset. Adjust if your preprocessing is different.
MODEL_DIM = 128 # The embedding dimension for the transformer
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
NUM_HEADS = 8 # Number of attention heads
DIM_FEEDFORWARD = 512
LEARNING_RATE = 1e-4
EPOCHS = 25
DROPOUT = 0.1

