import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger

import src.models.models as models
from src.data.datamodule import MyDataModule
from src.models.lightning_module import RainfallRegressionModel
from src.utils.config import load_config
from src.data.transforms import StandardScalerX, Log1pY

import argparse
import os
import uuid
import yaml
import inspect

def main(config):
    project_dir = os.path.join(config['logging']['base_dir'], config['logging']['project'])
    os.makedirs(project_dir, exist_ok=True)

    run_id = uuid.uuid4().hex[:8]
    out_dir = os.path.join(project_dir, run_id)
    os.makedirs(out_dir, exist_ok=True)

    wandb_logger = WandbLogger(
        project=config['logging']['project'],
        entity=config['logging'].get('entity'),
        name=config['logging'].get('run_name'),
        id=run_id,
        version=run_id,
        config=config,
        save_dir=out_dir
    )

    print(f"W&B run id: {run_id}")
    print(f"W&B run url: {wandb_logger.experiment.url}")

    # Persist the exact run config next to logs/checkpoints for reproducibility.
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    datamodule = MyDataModule(
        data_in_name=config['data']['data_in_name'],
        data_out_name=config['data']['data_out_name'],
        batch_size=config['data']['batch_size'],
        transform_X=StandardScalerX(),
        transform_y=None,
    )

    image_size = datamodule.image_shape[1]*datamodule.image_shape[2]

    model_name = config['model']['model_name']
    model_class = models.__dict__[model_name]

    
    model_config = {**config['model'], 'input_size': image_size, 'target_size': 1}
    constructor_params = inspect.signature(model_class.__init__).parameters
    allowed_kwargs = {
        name: value
        for name, value in model_config.items()
        if name in constructor_params and name != 'self'
    }
    model = model_class(**allowed_kwargs)
    
    lightning_model = RainfallRegressionModel(
        model,
        learning_rate=config['trainer']['learning_rate'],
        target_inverse_transform=None,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(out_dir, "checkpoints"),
        filename="best",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min"
    )

    trainer = L.Trainer(
        max_epochs=config['trainer']['max_epochs'],
        logger=wandb_logger,
        default_root_dir=out_dir,
        callbacks=[checkpoint_callback, early_stopping_callback ],
    )
    
    trainer.fit(lightning_model, datamodule=datamodule)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a rainfall regression model.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    args = parser.parse_args()
    config = load_config(args.config)
    main(config)
