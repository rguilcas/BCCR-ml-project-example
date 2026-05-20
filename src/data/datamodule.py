import lightning as L
import torch
from torch.utils.data import random_split, DataLoader
from .dataset import CustomDataset

# Note - you must have torchvision installed for this example


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
        self.image_shape = self.dataset_full[0][0].shape

    def setup(self, stage='fit'):
        n_total = len(self.dataset_full)
        n_train = int(self.train_share * n_total)
        n_val = int(self.val_share * n_total)
        n_test = n_total - n_train - n_val

        dataset_train, dataset_val, dataset_test = random_split(
            self.dataset_full,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42),
        )

        if stage in ('fit', 'validate', None):
            self.dataset_train = dataset_train
            self.dataset_val = dataset_val

        if stage in ('test', None):
            self.dataset_test = dataset_test

        if stage == 'predict':
            self.dataset_predict = dataset_test

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.dataset_predict, batch_size=self.batch_size)