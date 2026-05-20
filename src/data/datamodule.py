import lightning as L
import torch
from torch.utils.data import random_split, DataLoader, Subset
from .dataset import CustomDataset

# Note - you must have torchvision installed for this example

random_seed = 42

class CustomDataModule(L.LightningDataModule):
    def __init__(self, data_in_path, data_target_path, 
                 batch_size=32,
                 train_share=0.8, val_share=0.1,):
        super().__init__()
        self.data_in_path = data_in_path
        self.data_target_path = data_target_path
        self.batch_size = batch_size
        self.train_share = train_share
        self.val_share = val_share
        self.dataset_full = CustomDataset(self.data_in_path, self.data_target_path)
        self.predict_data = self.dataset_full
        self.image_shape = self.dataset_full[0]["x"].shape
    
    def setup(self, stage='fit'):
        n_total = len(self.dataset_full)
        n_train = int(self.train_share * n_total)
        n_val = int(self.val_share * n_total)
        n_test = n_total - n_train - n_val
        
        # Data splitting, we keep a sequential split now, but we can choose to shuffle them.

        train_indices = list(range(0, n_train))
        val_indices = list(range(n_train, n_train + n_val))
        test_indices = list(range(n_train + n_val, n_total))

        self.dataset_train = Subset(self.dataset_full, train_indices)
        self.dataset_val = Subset(self.dataset_full, val_indices)
        self.dataset_test = Subset(self.dataset_full, test_indices)
        self.times = self.dataset_full.times
        self.train_times = self.times[train_indices]
        self.val_times = self.times[val_indices]
        self.test_times = self.times[test_indices]


    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, generator=torch.Generator().manual_seed(random_seed))
    # We still shuffle the training data.

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.dataset_full, batch_size=self.batch_size)