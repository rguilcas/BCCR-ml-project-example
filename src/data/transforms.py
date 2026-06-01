import torch

class StandardScalerX:
    def __init__(self, eps=1e-6):
        self.meanX = None
        self.stdX = None
        self.fitted = False
        self.eps = eps

    def fit(self, dataset):
        X = torch.stack([dataset[i]['x'] for i in range(len(dataset))])
        self.meanX = X.mean(dim=0)
        self.stdX = X.std(dim=0, unbiased=False)
        self.stdX = torch.clamp(self.stdX, min=self.eps)

        self.fitted = True

    def __call__(self, x):
        if not self.fitted:
            raise RuntimeError("Scaler not fitted")
        return (x - self.meanX) / self.stdX



class MinMaxScalerY:
    def __init__(self, eps=1e-6):
        self.maxY = None
        self.minY = None
        self.fitted = False
        self.eps = eps

    def fit(self, dataset):
        Y = torch.stack([dataset[i]['y'] for i in range(len(dataset))])
        self.maxY = Y.max(dim=0).values
        self.minY = Y.min(dim=0).values
        self.fitted = True

    def inverse_transform(self, y):
        if not self.fitted:
            raise RuntimeError("Scaler not fitted")
        return y * (self.maxY - self.minY + self.eps) + self.minY

    def __call__(self, y):
        if not self.fitted:
            raise RuntimeError("Scaler not fitted")
        return (y - self.minY) / (self.maxY - self.minY + self.eps)
    

class Log1pY:
    def __init__(self, eps=1e-6):
        self.fitted = True

    def fit(self, dataset):
        pass

    def inverse_transform(self, y):
        return torch.expm1(y)
    def __call__(self, y):
        return torch.log1p(y)