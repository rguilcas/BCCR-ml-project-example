import lightning as L
import torch.nn as nn
import torch
from sklearn.metrics import r2_score

class RainfallRegressionModel(L.LightningModule):
    def __init__(self, model, learning_rate=1e-3):
        super().__init__()
        self.model = model
        self.criterion = nn.MSELoss()
        self.learning_rate = learning_rate
        self.val_preds = []
        self.val_targets = []

    def forward(self, x):
        return self.model(x)   
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.float()
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.float()
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.val_preds.append(y_hat.detach())
        self.val_targets.append(y.detach())
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss


    def on_validation_epoch_start(self):
        self.val_preds = []
        self.val_targets = []

    def on_validation_epoch_end(self):
        if not self.val_preds:
            return
        y_pred = torch.cat(self.val_preds, dim=0)
        y_true = torch.cat(self.val_targets, dim=0)
        r2 = r2_score(y_true.cpu().numpy(), y_pred.cpu().numpy())
        self.log('val_r2', r2, prog_bar=True, logger=True)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.float()
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
        x = x.float()
        return self(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
    