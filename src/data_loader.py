import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from glob import glob
from tqdm import tqdm
import config
from utils import create_sequences

def load_and_preprocess_data():
    """
    Main function to orchestrate the data loading and preprocessing pipeline
    as described in the interim report.
    """
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    
    # Check if final processed file exists
    benign_sequences_path = os.path.join(config.PROCESSED_DATA_DIR, 'X_benign_sequences.npy')
    if os.path.exists(benign_sequences_path):
        print(f"Preprocessed data found at {benign_sequences_path}. Loading...")
        return np.load(benign_sequences_path)

    # 1. Load and Concatenate CSV files
    csv_files = glob(os.path.join(config.DATA_DIR, '*.csv'))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {config.DATA_DIR}. Please add the CIC-DDoS2019 dataset.")

    df_list = [pd.read_csv(f) for f in tqdm(csv_files, desc="Loading CSVs")]
    df = pd.concat(df_list, ignore_index=True)

    # 2. Initial Cleaning
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Drop non-numeric or unnecessary columns for this model
    df = df.drop(columns=['Unnamed: 0', 'Flow ID', 'SimillarHTTP'], errors='ignore')
    
    # 3. Feature Engineering
    # Hash-encode IP addresses
    df['Source_IP_Hashed'] = df['Source IP'].apply(lambda x: hash(x))
    df['Destination_IP_Hashed'] = df['Destination IP'].apply(lambda x: hash(x))

    # Convert Timestamp to Time_Since_Start
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Time_Since_Start'] = (df['Timestamp'] - df['Timestamp'].min()).dt.total_seconds()
    
    # Drop original columns that have been engineered
    df = df.drop(columns=['Source IP', 'Destination IP', 'Timestamp'])

    # 4. Benign Data Isolation
    benign_df = df[df['Label'] == 'BENIGN'].copy()
    
    # Keep only numerical features for the model
    numerical_cols = benign_df.select_dtypes(include=np.number).columns.tolist()
    benign_df = benign_df[numerical_cols]

    print(f"Isolated {len(benign_df)} benign samples.")
    print(f"Number of features before scaling: {len(benign_df.columns)}")

    # 5. Final Scaling and Sequencing
    scaler = StandardScaler()
    benign_scaled = scaler.fit_transform(benign_df)
    
    # Create sequences
    X_benign_sequences = create_sequences(benign_scaled, config.SEQUENCE_LENGTH)

    print(f"Final training data shape: {X_benign_sequences.shape}") # Should be (56326, 100, 86) as per report
    
    # Save the processed data
    np.save(benign_sequences_path, X_benign_sequences)
    print(f"Processed benign sequences saved to {benign_sequences_path}")
    
    return X_benign_sequences

if __name__ == '__main__':
    load_and_preprocess_data()
