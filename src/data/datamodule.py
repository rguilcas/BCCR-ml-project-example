import os
import lightning as L
import torch
import xarray as xr
import numpy as np
from torch.utils.data import random_split, DataLoader, Subset
from src.data.dataset import AtmosphereToRainfallDataset

random_seed = 42

class MyDataModule(L.LightningDataModule):
        # Data splitting, we keep a sequential split now, but we can choose to shuffle them.
    def __init__(self, data_in_name, data_target_name, 
                 data_path='./data/raw',
                 batch_size=32,
                 years_split = [('1979','2008'), ('2009','2014'), ('2015','2024')],
                 transform_X=None, transform_y=None):
        super().__init__()
        self.data_in_name = data_in_name
        self.data_target_name = data_target_name
        self.data_path = data_path
        self.batch_size = batch_size
        self.train_years, self.val_years, self.test_years = years_split
        self.transform_X = transform_X
        self.transform_y = transform_y
        self._transforms_fitted = False
        with xr.open_dataarray(os.path.join(data_path, data_in_name)) as da:
            self.image_shape = tuple(da.isel(time=0).shape)
        self.image_size = int(np.prod(self.image_shape[-2:]))

    def setup(self, stage='fit'):
        # Training phase
        if stage == 'fit':
            self.dataset_train = AtmosphereToRainfallDataset(
                self.data_in_name, 
                self.data_target_name, 
                data_path=self.data_path,
                time_slice=self.train_years)
            self.dataset_val = AtmosphereToRainfallDataset(
                self.data_in_name, 
                self.data_target_name, 
                data_path=self.data_path,
                time_slice=self.val_years)
            # Data preprocessing, we fit the transforms on the training data and then apply them to train, val and test. 
            if not self._transforms_fitted:
                if self.transform_X is not None:
                    self.transform_X.fit(self.dataset_train)
                if self.transform_y is not None:
                    self.transform_y.fit(self.dataset_train)
                self._transforms_fitted = True
            self.dataset_train.transform_X = self.transform_X
            self.dataset_val.transform_X = self.transform_X
            self.dataset_train.transform_y = self.transform_y
            self.dataset_val.transform_y = self.transform_y
            
        # Testing phase
        if stage == 'test':
            self.dataset_test = AtmosphereToRainfallDataset(self.data_in_name, self.data_target_name, 
                                                            data_path=self.data_path,time_slice=self.test_years)
            self.dataset_test.transform_X = self.transform_X
            self.dataset_test.transform_y = self.transform_y
        
        # Prediction phase
        if stage == 'predict':
            self.dataset_full = AtmosphereToRainfallDataset(self.data_in_name, self.data_target_name, 
                                                         data_path=self.data_path)
            self.dataset_full.transform_X = self.transform_X
            self.dataset_full.transform_y = self.transform_y
        
        

    def train_dataloader(self):
        # We still shuffle the training data.
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, generator=torch.Generator().manual_seed(random_seed))
    
    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.dataset_full, batch_size=self.batch_size)