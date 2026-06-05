import lightning as L
from lightning.pytorch.loggers import WandbLogger

import src.models.models as models
from src.data.datamodule import MyDataModule
from src.models.lightning_module import RegressionModel
from src.utils.config import load_config
from src.data import transforms 
from src.training.callbacks import (
    get_checkpoint_callback,
    get_early_stopping_callback,
    get_validation_plots_callback,
)

import argparse
import os
import uuid
import yaml
import inspect
import numpy as np

def main(config):
    project_dir = os.path.join(config['logging']['base_dir'], config['logging']['project'])
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

    # Save config file for reproducibility
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    if config['data']['transform_X'] != 'None':
        transform_X = getattr(transforms, config['data']['transform_X'])()
    else:
        transform_X = None

    if config['data']['transform_y'] != 'None':
        transform_y = getattr(transforms, config['data']['transform_y'])()
    else:
        transform_y = None


    datamodule = MyDataModule(
        data_in_name=config['data']['data_in_name'],
        data_target_name=config['data']['data_target_name'],
        data_path=config['data']['data_path'],
        batch_size=config['data']['batch_size'],
        transform_X=transform_X,
        transform_y=transform_y,
    )
    
    model_class = getattr(models, config["model"]["model_name"])
    model_config = {
		**config["model"],
		"image_size": datamodule.image_size,
		"n_channels_input_cnn": config["model"]["n_channels_input_cnn"],
		"input_size": datamodule.image_size * config["model"]["n_channels_input_cnn"],
		"target_size": 1,
	}
    constructor_params = inspect.signature(model_class.__init__).parameters
    allowed_kwargs = {
		name: value
		for name, value in model_config.items()
		if name in constructor_params and name != "self"
	}
    neural_network = model_class(**allowed_kwargs)

    target_inverse_transform = None
    if transform_y is not None and hasattr(transform_y, 'inverse_transform'):
        target_inverse_transform = transform_y.inverse_transform
    
    lightning_model = RegressionModel(
        neural_network,
        learning_rate=config['trainer']['learning_rate'],
        target_inverse_transform=target_inverse_transform,
    )

    trainer = L.Trainer(
        max_epochs=config['trainer']['max_epochs'],
        accelerator=config['trainer']['accelerator'],
        devices=config['trainer']['devices'],
        logger=wandb_logger,
        default_root_dir=out_dir,
        callbacks=[
            get_checkpoint_callback(out_dir),
            get_early_stopping_callback(),
            # get_validation_plots_callback(),
        ],
    )
    
    trainer.fit(lightning_model, datamodule=datamodule)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a rainfall regression model.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    args = parser.parse_args()
    config = load_config(args.config)
    main(config)



