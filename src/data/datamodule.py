import lightning as L
import torch
from torch.utils.data import random_split, DataLoader, Subset
from src.data.dataset import AtmosphereToRainfallDataset

random_seed = 42

class MyDataModule(L.LightningDataModule):
    def __init__(self, data_in_name, data_target_name, 
                 batch_size=32,
                 train_share=0.8, val_share=0.1,
                 transform_X=None, transform_y=None):
        super().__init__()
        self.data_in_name = data_in_name
        self.data_target_name = data_target_name
        self.batch_size = batch_size
        self.train_share = train_share
        self.val_share = val_share
        self.transform_X = transform_X
        self.transform_y = transform_y
        self.dataset_full = AtmosphereToRainfallDataset(self.data_in_name, self.data_target_name, transform_X=self.transform_X, transform_y=self.transform_y)
        self.predict_data = self.dataset_full
        self.image_shape = self.dataset_full[0]["x"].shape
        self._transformX_applied = False
        self._transformY_applied = False
    
    def prepare_data(self):
        return 
            
    def setup(self, stage='fit'):
        # Data splitting, we keep a sequential split now, but we can choose to shuffle them here.
        n_total = len(self.dataset_full)
        n_train = int(self.train_share * n_total)
        n_val = int(self.val_share * n_total)
        
        train_indices = list(range(0, n_train))
        val_indices = list(range(n_train, n_train + n_val))
        test_indices = list(range(n_train + n_val, n_total))

        self.dataset_train = Subset(self.dataset_full, train_indices)
        self.dataset_val = Subset(self.dataset_full, val_indices)
        self.dataset_test = Subset(self.dataset_full, test_indices)
        if self.transform_X is not None and not self._transformX_applied:
            self.transform_X.fit(self.dataset_train)
            # All subsets share dataset_full, so apply transform once on the base dataset.
            self.dataset_full.transform_X = self.transform_X
            self._transformX_applied = True
        if self.transform_y is not None and not self._transformY_applied:
            self.transform_y.fit(self.dataset_train)
            self.dataset_full.transform_y = self.transform_y
            self._transformY_applied = True

        # We keep the times for each subset, which will be used for evaluation and visualization.
        self.times = self.dataset_full.times
        self.train_times = self.times[train_indices]
        self.val_times = self.times[val_indices]
        self.test_times = self.times[test_indices]


    def train_dataloader(self):
        # We still shuffle the training data.
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, generator=torch.Generator().manual_seed(random_seed))
    

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.dataset_full, batch_size=self.batch_size)