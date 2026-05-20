from torch.utils.data import Dataset
import torch
import pandas as pd
import os 
import xarray as xr
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, ds_in_path, ds_target_path, 
                 transform=None, target_transform=None):
        # Load eagerly and close file handles to avoid multiprocessing shutdown issues.
        with xr.open_dataarray(ds_in_path) as ds_in_da:
            ds_in = ds_in_da.load()
        with xr.open_dataarray(ds_target_path) as ds_target_da:
            ds_target = ds_target_da.load()
        common_times = np.intersect1d(ds_in.time.values, ds_target.time.values)
        # Select only the common times for both datasets and choose the appropriate split
        self.ds_in = ds_in.sel(time=common_times)   
        self.ds_target = ds_target.sel(time=common_times)

        self.transform = transform
        self.target_transform = target_transform
        self.times = self.ds_in.time.values

    def __len__(self):
        # Determines the number of samples in the dataset, which is the length of the time dimension after filtering for common times and splits.
        return len(self.times)

    def __getitem__(self, idx):
        # Determines how one sample is retrieved. Here we select the appropriate time step from both datasets and apply any transformations if specified.
        x = self.ds_in.isel(time=[idx]).values
        y = self.ds_target.isel(time=[idx]).values
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        x = torch.as_tensor(x, dtype=torch.float32)
        y = torch.as_tensor(y, dtype=torch.float32)
        return x, y
