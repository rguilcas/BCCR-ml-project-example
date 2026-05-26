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


class Log1pY:
    def fit(self, dataset):
        return

    def __call__(self, y):
        return torch.log1p(torch.clamp(y, min=0.0)) 

    def inverse(self, y_transformed):
        return torch.expm1(y_transformed)