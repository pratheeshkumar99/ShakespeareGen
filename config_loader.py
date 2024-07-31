import json
import torch
import sys

def get_device():
    """Determines the best available device for computation."""
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

def load_config(file_path='config.json'):
    """
    Loads configuration settings from a JSON file and sets the appropriate device.

    Args:
    file_path (str): Path to the JSON configuration file.

    Returns:
    dict: A dictionary containing configuration parameters.
    """
    try:
        with open(file_path, 'r') as file:
            config = json.load(file)
    except FileNotFoundError:
        print("Configuration file not found.", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print("Error decoding the configuration file.", file=sys.stderr)
        sys.exit(1)

    # Set the device in the configuration based on the system's capabilities
    config['model']['device'] = get_device()
    return config

if __name__ == '__main__':
    config = load_config()
    print(config)
    print(f"Device: {config['model']['device']}")