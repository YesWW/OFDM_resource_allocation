import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import dill
from tqdm import tqdm

class ImitationDataset(Dataset):
    def __init__(self, trajectory_file, simulator, min_attn_db=-200, max_attn_db=-50, device='cpu'):
        with open(trajectory_file, 'rb') as f:
            self.data = dill.load(f)
        self.graphs = simulator.generate_pyg(
            networks=[item['network'] for item in self.data],
            min_attn_db=min_attn_db, max_attn_db=max_attn_db, device=device
        )
        self.actions = [item['action'] for item in self.data]

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        action = self.actions[idx]
        label_power = torch.tensor(action['power_level'], dtype=torch.long)  # shape: (link, RB)
        label_beam = torch.tensor(action['beam_index'], dtype=torch.long)   # shape: (link, RB)
        return graph, label_power, label_beam