import torch
from model import TextGeneratorModel
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config_loader import load_config
import argparse

def load_model(model_path, config):
    """
    Loads the model from the specified path with the given configuration.
    """

    model = TextGeneratorModel(config["model"]).to(config["model"]["device"])
    try:
        model.load_state_dict(torch.load(model_path))
        model.eval()
    except Exception as e:
        print(f"Failed to load model: {e}", file=sys.stderr)
        sys.exit(1)
    return model

def load_char_mappings(mapping_path):
    """
    Load character mappings (stoi and itos).
    """
    try:
        mappings = torch.load(mapping_path)
    except Exception as e:
        print(f"Failed to load mappings: {e}", file=sys.stderr)
        sys.exit(1)
    return mappings['stoi'], mappings['itos']

def generate_text(model, itos, max_len=100):
    """
    Generates text starting from a zero-initialized index tensor.
    """
    idx_tensor = torch.zeros((1, 1), dtype=torch.long, device=model.config["device"])
    model.to(model.config["device"])  # Ensure model is on the correct device
    
    try:
        output = model.generate_text(idx_tensor, max_len)
        generated_text = ''.join([itos[idx] for idx in output[0].tolist()])
        print("Generated Text:", generated_text)
    except RuntimeError as e:
        print(f"Caught an error during generation: {e}", file=sys.stderr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate text using a trained model')
    parser.add_argument('max_len', type=int, help='Maximum number of tokens to generate')
    args = parser.parse_args()

    config = load_config()  # Ensure this function loads and sets the device correctly
    model_path = os.path.join('..', 'models', 'text_generator_model_latest.pth')  # Adjust the path as necessary
    mapping_path = os.path.join('..', 'models', 'char_mappings.pth')

    model = load_model(model_path, config)
    _, itos = load_char_mappings(mapping_path)
    
    generate_text(model, itos, max_len=args.max_len)