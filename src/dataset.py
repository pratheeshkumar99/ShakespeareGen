import torch
from torch.utils.data import Dataset

class CharDataset(Dataset):
    """Character dataset for NLP model training.

    This class prepares chunks of data for training a model based on character sequences.

    Attributes:
        data (tensor): The dataset composed of numerical indices of characters.
        block_size (int): The size of a sequence of characters to process in one go.
    """

    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        """Retrieves a sample from the dataset at the specified index."""
        chunk = self.data[idx:idx + self.block_size + 1]
        return chunk[:-1], chunk[1:]