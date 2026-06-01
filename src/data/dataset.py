import os

from torch.utils.data import Dataset
import torch
import pandas as pd
import xarray as xr
import numpy as np

class AtmosphereToRainfallDataset(Dataset):
    def __init__(self, 
                 data_in_name='msl_data.nc', 
                 data_target_name='pr_era5_daily_westnorway_1940-2024.nc', 
                 transform_X=None, transform_y=None,
                 data_path = './data/raw',
                 time_slice = (None,None)):
        # Load files with context manager to prevent memory issues.
        with xr.open_dataarray(os.path.join(data_path, data_in_name)) as ds_in_da:
            ds_in = ds_in_da.load()
        with xr.open_dataarray(os.path.join(data_path, data_target_name)) as ds_target_da:
            ds_target = ds_target_da.load()
        if len(ds_in.shape) == 3:
            # Create a feature channel if the input data is 3D (time, lat, lon) to make it compatible with CNNs.
            ds_in = ds_in.expand_dims('feature', axis=1)
        common_times = np.intersect1d(ds_in.time.values, ds_target.time.values)
        # Select only the common times for both datasets and choose the appropriate split
        ds_total = xr.Dataset({'predictors': ds_in, 'targets': ds_target}).sel(time=common_times)
        self.ds = ds_total.sel(time=common_times).sel(time=slice(time_slice[0], time_slice[1]))
        self.transform_X = transform_X
        self.transform_y = transform_y
        self.times = self.ds.time.values

    def __len__(self):
        # Determines the number of samples in the dataset, which is the length of the time dimension after filtering for common times and splits.
        return self.times.size

    def __getitem__(self, idx):
        # Determines how one sample is retrieved. Here we select the appropriate time step from the dataset and apply any transformations if specified.
        x = torch.as_tensor(self.ds.predictors.isel(time=idx).values, dtype=torch.float32)
        y = torch.as_tensor(self.ds.targets.isel(time=idx).values, dtype=torch.float32)
        # Keep target as a 1D tensor of length 1 so batched targets are [B, 1].
        if y.ndim == 0:
            y = y.unsqueeze(0)
        if self.transform_X:
            x = self.transform_X(x)
        if self.transform_y:
            y = self.transform_y(y)
        return {"x": x, "y": y, "idx": idx}