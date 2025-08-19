import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from glob import glob
from tqdm import tqdm
import joblib
import config
from utils import create_sequences

def get_benign_dataframe_from_csvs():
    """
    Reads all raw CSVs, filters for benign traffic, and returns a single
    DataFrame. This is memory-intensive but only needs to be run once to create
    the benign cache.
    """
    if os.path.basename(os.getcwd()) == 'notebooks':
        base_path = '..'
    else:
        base_path = '.'
    
    csv_files = glob(os.path.join(base_path, config.DATA_DIR, '*.csv'))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {os.path.join(base_path, config.DATA_DIR)}.")

    benign_chunks = []
    for f in tqdm(csv_files, desc="Scanning CSVs for benign data"):
        try:
            if 'BENIGN' in pd.read_csv(f, usecols=[' Label'])[' Label'].unique():
                full_chunk = pd.read_csv(f)
                benign_part = full_chunk[full_chunk[' Label'] == 'BENIGN']
                if not benign_part.empty:
                    benign_chunks.append(benign_part)
        except Exception as e:
            print(f"Warning: Could not process file {f}. Error: {e}")
            continue
            
    if not benign_chunks:
        raise ValueError("No benign data found in any of the CSV files.")
        
    return pd.concat(benign_chunks, ignore_index=True)


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

    print("Cached benign data/scaler not found. Processing from raw CSVs...")
    benign_df = get_benign_dataframe_from_csvs()
    
    benign_df.columns = benign_df.columns.str.strip()
    benign_df = benign_df.drop(columns=['Unnamed: 0', 'Flow ID', 'SimillarHTTP'], errors='ignore')
    benign_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    benign_df.dropna(inplace=True)

    benign_df['Source_IP_Hashed'] = benign_df['Source IP'].apply(lambda x: hash(x))
    benign_df['Destination_IP_Hashed'] = benign_df['Destination IP'].apply(lambda x: hash(x))
    benign_df['Timestamp'] = pd.to_datetime(benign_df['Timestamp'])
    benign_df['Time_Since_Start'] = (benign_df['Timestamp'] - benign_df['Timestamp'].min()).dt.total_seconds()
    
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

def create_and_cache_all_attack_sets(samples_per_type=100000):
    """
    Scans the raw CSVs and creates a separate, sampled .npy cache file for
    each attack type found. Skips any attack type that fails.
    """
    print("Starting creation of individual attack sequence caches...")
    
    base_path = '.'
    scaler_path = os.path.join(base_path, config.SCALER_SAVE_PATH)
    csv_files = glob(os.path.join(base_path, config.DATA_DIR, '*.csv'))

    if not csv_files:
        raise FileNotFoundError("Raw CSV files not found.")

    try:
        scaler = joblib.load(scaler_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Scaler not found at {scaler_path}. Run benign preprocessing first.")

    print("Determining all available attack labels...")
    all_labels = set()
    for f in tqdm(csv_files, desc="Scanning for labels"):
        try:
            all_labels.update(pd.read_csv(f, usecols=[' Label'])[' Label'].unique())
        except Exception:
            continue
    
    attack_labels = [label for label in all_labels if label != 'BENIGN']
    print(f"Found {len(attack_labels)} attack types.")

    for label in attack_labels:
        output_path = os.path.join(base_path, config.PROCESSED_DATA_DIR, f'X_attack_{label}.npy')
        if os.path.exists(output_path):
            print(f"Cache for '{label}' already exists. Skipping.")
            continue
            
        print(f"\nProcessing attack type: {label}")
        try:
            attack_samples_df_list = []
            for f in tqdm(csv_files, desc=f"Reading '{label}'"):
                try:
                    for chunk_df in pd.read_csv(f, chunksize=100000, low_memory=False):
                        attack_chunk = chunk_df[chunk_df[' Label'] == label]
                        if not attack_chunk.empty:
                            attack_samples_df_list.append(attack_chunk)
                except Exception:
                    continue
            
            if not attack_samples_df_list:
                print(f"No samples found for {label}. Skipping.")
                continue
            
            full_attack_df = pd.concat(attack_samples_df_list, ignore_index=True)
            
            if len(full_attack_df) > samples_per_type:
                full_attack_df = full_attack_df.sample(n=samples_per_type, random_state=42)

            full_attack_df.columns = full_attack_df.columns.str.strip()
            full_attack_df = full_attack_df.drop(columns=['Unnamed: 0', 'Flow ID', 'SimillarHTTP'], errors='ignore')
            full_attack_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            full_attack_df.dropna(inplace=True)
            full_attack_df['Source_IP_Hashed'] = full_attack_df['Source IP'].apply(lambda x: hash(x))
            full_attack_df['Destination_IP_Hashed'] = full_attack_df['Destination IP'].apply(lambda x: hash(x))
            full_attack_df['Timestamp'] = pd.to_datetime(full_attack_df['Timestamp'])
            full_attack_df['Time_Since_Start'] = (full_attack_df['Timestamp'] - pd.Timestamp.now()).dt.total_seconds()
            
            numerical_cols = full_attack_df.select_dtypes(include=np.number).columns.tolist()
            df_numerical = full_attack_df[numerical_cols].copy()
            df_numerical = df_numerical.reindex(columns=scaler.feature_names_in_, fill_value=0)
            
            scaled_data = scaler.transform(df_numerical)
            sequences = create_sequences(scaled_data, config.SEQUENCE_LENGTH)
            
            if sequences.shape[0] > 0:
                np.save(output_path, sequences)
                print(f"Successfully created cache for {label} with {sequences.shape[0]} sequences.")
            else:
                print(f"Could not generate sequences for {label}.")

        except Exception as e:
            print(f"!!! FAILED to create cache for '{label}'. Error: {e}. Skipping. !!!")
            continue

if __name__ == '__main__':
    print("--- Running Benign Data Preprocessing ---")
    load_and_preprocess_data()
    print("\n--- Running Attack Data Caching ---")
    create_and_cache_all_attack_sets()
