import lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from src.models.model import SimpleMLP
from src.data.datamodule import CustomDataModule
from src.models.lightning_module import RainfallRegressionModel
from src.utils.config import load_config
import argparse
import os

def main(config):

    wandb_logger = WandbLogger(
        project=config['logging']['project'],
        name=config['logging'].get('run_name'),
        config=config,
        save_dir=config['logging']['base_dir']
    )

    run_id = wandb_logger.experiment.id
    out_dir = f"{config['logging']['base_dir']}/{config['logging']['project']}/{run_id}"

    datamodule = CustomDataModule(
        data_in_path=config['data']['input_path'],
        data_target_path=config['data']['target_path'],
        batch_size=config['data']['batch_size']
    )

    image_size = datamodule.image_shape[1]*datamodule.image_shape[2]

    model = SimpleMLP(input_size=image_size, 
                      hidden_size=config['model']['hidden_size'], 
                      target_size=1)
    
    lightning_model = RainfallRegressionModel(model, learning_rate=config['trainer']['learning_rate'])

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(out_dir, "checkpoints"),
        filename="best",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
    )

    trainer = L.Trainer(
        max_epochs=config['trainer']['max_epochs'],
        logger=wandb_logger,
        default_root_dir=out_dir,
        callbacks=[checkpoint_callback],
    )
    
    trainer.fit(lightning_model, datamodule=datamodule)
    trainer.validate(lightning_model, datamodule=datamodule)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a rainfall regression model.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    args = parser.parse_args()
    config = load_config(args.config)
    main(config)
