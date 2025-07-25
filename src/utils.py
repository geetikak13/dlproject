# src/utils.py

import logging
import sys

def setup_logger():
    """
    Sets up a basic logger to print information to the console.
    """
    logger = logging.getLogger("DDoS_Anomaly_Detection")
    logger.setLevel(logging.INFO)

    # Create a handler to write to stdout
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Add the handler to the logger
    if not logger.handlers:
        logger.addHandler(handler)

    return logger

# Example of another potential utility function
def save_config(config_module, file_path):
    """
    Saves the current configuration to a file for reproducibility.
    
    Args:
        config_module: The config module object.
        file_path (str): The path to save the configuration file.
    """
    with open(file_path, 'w') as f:
        f.write("# Project Configuration\n\n")
        for key, value in config_module.__dict__.items():
            if not key.startswith('__') and isinstance(value, (str, int, float, bool, list, dict)):
                f.write(f"{key} = {repr(value)}\n")
    print(f"Configuration saved to {file_path}")

