import argparse
from pathlib import Path

import lightning as L
import pandas as pd
import torch
import numpy as np
from src.data.datamodule import CustomDataModule
from src.models.lightning_module import RainfallRegressionModel
import src.models.models as models
from src.utils.config import load_config
import inspect


def predict(config, checkpoint_path, output_path):
	datamodule = CustomDataModule(
		data_in_path=config["data"]["input_path"],
		data_target_path=config["data"]["target_path"],
		batch_size=config["data"]["batch_size"],
	)
	# Setup for correct split
	image_size = datamodule.image_shape[1] * datamodule.image_shape[2]    
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
    

	lightning_model = RainfallRegressionModel.load_from_checkpoint(
		checkpoint_path,
		model=model,
		learning_rate=config["trainer"]["learning_rate"],
	)

	trainer = L.Trainer(logger=False)
	datamodule.setup(stage="predict")
	prediction_batches = trainer.predict(lightning_model, datamodule=datamodule)

	preds = torch.cat([batch[0].detach().cpu() for batch in prediction_batches], dim=0)
	targets = torch.cat([batch[1].detach().cpu() for batch in prediction_batches], dim=0)
	times = np.concat([datamodule.times[batch[2]] for batch in prediction_batches])
	# times = datamodule.times

	df = pd.DataFrame(
		{
			"prediction": preds.squeeze(-1).numpy(),
			"target": targets.squeeze(-1).numpy(),
		},
		index=times
	)
	df['data_slice'] = 'train'
	df.loc[datamodule.val_times,'data_slice'] = 'val'
	df.loc[datamodule.test_times,'data_slice'] = 'test'
	
	output_file = Path(output_path)
	output_file.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(output_file)
	
	print(f"Saved predictions to: {output_file}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Run inference from a trained checkpoint.")
	parser.add_argument(
		"--run-id",
		type=str,
		required=True,
		help="Name of the run to load a checkpoint from. Should correspond to a W&B run id.",
	)
	parser.add_argument(
		"--project-root",
		type=str,
		default='bccr-ml-project',
		help="Name of the W&B project where the run is logged.",
	)
	parser.add_argument(
		"--checkpoint-name",
		type=str,
		default="best.ckpt",
		help="Path to a .ckpt file produced during training.",
	)
	parser.add_argument(
		"--output",
		type=str,
		default=None,
		help="Path for the predictions to be saved.",    
	)
	parser.add_argument(
		"--split",
		type=str,
		default="val",
		choices=["val", "test"],
		help="Which split to predict: 'val' or 'test'",
	)
	args = parser.parse_args()
	config_path = f"outputs/{args.project_root}/{args.run_id}/config.yaml"
	checkpoint_path = f"outputs/{args.project_root}/{args.run_id}/checkpoints/{args.checkpoint_name}"
	if args.output is None:
		args.output = f"outputs/{args.project_root}/{args.run_id}/predictions.csv"
	config = load_config(config_path)
	predict(config=config, checkpoint_path=checkpoint_path, output_path=args.output)