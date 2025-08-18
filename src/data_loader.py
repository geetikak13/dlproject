import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from glob import glob
from tqdm import tqdm
import joblib
import config
from utils import create_sequences

def get_combined_dataframe():
    """Helper function to load and cache the combined dataframe."""
    # Adjust path for notebook execution
    if os.path.basename(os.getcwd()) == 'notebooks':
        base_path = '..'
    else:
        base_path = '.'
        
    feather_path = os.path.join(base_path, config.PROCESSED_DATA_DIR, 'combined_dataset.feather')
    if os.path.exists(feather_path):
        return pd.read_feather(feather_path)

    csv_files = glob(os.path.join(base_path, config.DATA_DIR, '*.csv'))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {os.path.join(base_path, config.DATA_DIR)}. Please add the CIC-DDoS2019 dataset.")

    df_list = [pd.read_csv(f) for f in tqdm(csv_files, desc="Loading CSVs")]
    df = pd.concat(df_list, ignore_index=True)
    
    # Basic cleaning
    df.columns = df.columns.str.strip()
    df = df.drop(columns=['Unnamed: 0', 'Flow ID', 'SimillarHTTP'], errors='ignore')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    # Feature Engineering
    df['Source_IP_Hashed'] = df['Source IP'].apply(lambda x: hash(x))
    df['Destination_IP_Hashed'] = df['Destination IP'].apply(lambda x: hash(x))
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Time_Since_Start'] = (df['Timestamp'] - df['Timestamp'].min()).dt.total_seconds()
    
    # Save to feather for faster loading next time
    os.makedirs(os.path.dirname(feather_path), exist_ok=True)
    df.to_feather(feather_path)
    return df

def load_and_preprocess_data():
    """Loads and preprocesses the BENIGN data for training."""
    if os.path.basename(os.getcwd()) == 'notebooks':
        base_path = '..'
    else:
        base_path = '.'

    benign_sequences_path = os.path.join(base_path, config.PROCESSED_DATA_DIR, 'X_benign_sequences.npy')
    scaler_path = os.path.join(base_path, config.SCALER_SAVE_PATH)

    if os.path.exists(benign_sequences_path) and os.path.exists(scaler_path):
        print(f"Preprocessed benign data and scaler found. Loading from cache...")
        return np.load(benign_sequences_path)

    print("Cached data/scaler not found or incomplete. Running full preprocessing for benign data...")
    df = get_combined_dataframe()
    benign_df = df[df['Label'] == 'BENIGN'].copy()
    
    numerical_cols = benign_df.select_dtypes(include=np.number).columns.tolist()
    benign_df_numerical = benign_df[numerical_cols].copy()

    scaler = StandardScaler()
    benign_scaled = scaler.fit_transform(benign_df_numerical)
    
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    
    X_benign_sequences = create_sequences(benign_scaled, config.SEQUENCE_LENGTH)
    np.save(benign_sequences_path, X_benign_sequences)
    print(f"Processed benign sequences saved to {benign_sequences_path}")
    
    return X_benign_sequences

def load_and_preprocess_attack_data(attack_label: str):
    """Loads and preprocesses a specific ATTACK data type for analysis."""
    print(f"Loading and preprocessing data for attack type: {attack_label}")
    
    if os.path.basename(os.getcwd()) == 'notebooks':
        base_path = '..'
    else:
        base_path = '.'
    
    scaler_path = os.path.join(base_path, config.SCALER_SAVE_PATH)
    
    try:
        scaler = joblib.load(scaler_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Scaler not found at {scaler_path}. "
                                "Please run the benign data preprocessing first by running `train.py` or `data_loader.py`.")

    df = get_combined_dataframe()
    attack_df = df[df['Label'] == attack_label].copy()

    if attack_df.empty:
        raise ValueError(f"No data found for attack label '{attack_label}'. "
                         f"Available labels include: {df['Label'].unique()}")

    # --- Memory Optimization ---
    # If the attack dataframe is very large, take a manageable sample to prevent memory errors.
    # 50,000 is more than enough for a robust evaluation.
    max_samples = 50000
    if len(attack_df) > max_samples:
        print(f"Attack dataset is very large ({len(attack_df)} rows). Taking a random sample of {max_samples} rows.")
        attack_df = attack_df.sample(n=max_samples, random_state=42)

    numerical_cols = attack_df.select_dtypes(include=np.number).columns.tolist()
    attack_df_numerical = attack_df[numerical_cols].copy()
    
    scaler_features = scaler.feature_names_in_
    attack_df_numerical = attack_df_numerical.reindex(columns=scaler_features, fill_value=0)

    attack_scaled = scaler.transform(attack_df_numerical)
    
    X_attack_sequences = create_sequences(attack_scaled, config.SEQUENCE_LENGTH)
    
    if X_attack_sequences.shape[0] == 0:
        raise ValueError(f"Not enough data for '{attack_label}' to create a sequence of length {config.SEQUENCE_LENGTH}.")

    print(f"Created {X_attack_sequences.shape[0]} sequences for '{attack_label}'.")
    return X_attack_sequences

if __name__ == '__main__':
    load_and_preprocess_data()
