import numpy as np
from tqdm import tqdm

def create_sequences(data: np.ndarray, sequence_length: int):
    """
    Creates overlapping sequences from the input data.
    """
    sequences = []
    data_len = len(data)
    print(f"Creating sequences of length {sequence_length}...")
    for i in tqdm(range(data_len - sequence_length + 1)):
        seq = data[i:(i + sequence_length)]
        sequences.append(seq)
    return np.array(sequences)
