import torch

# -- Data Paths --
DATA_DIR = "data/"
PROCESSED_DATA_DIR = "data/processed/"
MODEL_SAVE_PATH = "saved_models/transformer_autoencoder.pth"
SCALER_SAVE_PATH = "saved_models/scaler.joblib"
RESULTS_DIR = "results/"

# -- Data Processing Parameters --
SEQUENCE_LENGTH = 100  # Length of input sequences

# -- Model Hyperparameters --
INPUT_FEATURES = 86
MODEL_DIM = 128      # d_model in Transformer literature
NUM_HEADS = 8        # Number of attention heads
NUM_ENCODER_LAYERS = 3
DIM_FEEDFORWARD = 512
DROPOUT = 0.1

# -- Training Hyperparameters --
LEARNING_RATE = 0.0001
BATCH_SIZE = 64
NUM_EPOCHS = 20

# -- Device Configuration --
# Set the device to CUDA for NVIDIA GPUs, MPS for Apple Silicon, or CPU as a fallback.
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
