from torch.utils.data import DataLoader, Dataset
import torch

class ToyDataset(Dataset):
    def __init__(self, dataset_config):
        super(ToyDataset, self).__init__()
        h, w = dataset_config['image_size']
        self.fake_data = torch.randn((3, h, w))
        self.fake_target = torch.tensor([
            [0, 0, 1, 1, 2, 2]
        ])

    def __len__(self):
        return 640

    def __getitem__(self, idx):
        return self.fake_data, self.fake_target