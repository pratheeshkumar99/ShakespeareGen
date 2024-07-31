import torch
from model import TextGeneratorModel
from dataset import CharDataset
from torch.utils.data import DataLoader
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config_loader import load_config
import matplotlib.pyplot as plt



def train_epoch(data_loader, model, optimizer, device):
    model.train()
    total_loss = 0
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        _, loss = model(inputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

@torch.no_grad()
def validate(data_loader, model, device):
    model.eval()
    total_loss = 0
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        _, loss = model(inputs, targets)
        total_loss += loss.item()
    return total_loss / len(data_loader)

def load_data():
    """Loads character data from the input file and prepares datasets."""
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'input.txt')
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encoded_data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(encoded_data))
    train_data, val_data = encoded_data[:n], encoded_data[n:]
    return train_data, val_data, len(chars), stoi, itos

def main():
    config = load_config()
    print("Configuration loaded successfully.")
    train_data, val_data, vocab_size, stoi, itos = load_data()
    device = config['model']['device']
    model = TextGeneratorModel(config['model']).to(device)
    print("Model created successfully.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['model']['lr'])

    train_dataset = CharDataset(train_data, config['model']['block_size'])
    val_dataset = CharDataset(val_data, config['model']['block_size'])
    train_loader = DataLoader(train_dataset, batch_size=config['model']['batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config['model']['batch_size'], shuffle=False, drop_last=True)

    num_epochs = 10
    train_losses, val_losses = [], []
    print("Training started.")
    for epoch in range(num_epochs):
        train_loss = train_epoch(train_loader, model, optimizer, device)
        val_loss = validate(val_loader, model, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f'Epoch {epoch + 1}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}')


    # Define the path to save the model
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)  # Create the directory if it doesn't exist

    model_path = os.path.join(model_dir, 'text_generator_model_latest.pth')
    torch.save(model.state_dict(), model_path)

    # Save character mappings
    mapping_path = os.path.join(model_dir, 'char_mappings.pth')
    torch.save({'stoi': stoi, 'itos': itos}, mapping_path)

    plot_path = os.path.join(model_dir, 'training_validation_loss_plot.png')

    plot_losses(train_losses, val_losses, plot_path)

def plot_losses(train_losses, val_losses, path='training_validation_loss_plot.png'):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='x')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(path)
    plt.close()

if __name__ == "__main__":
    main()