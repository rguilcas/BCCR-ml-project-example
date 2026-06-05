import argparse
import inspect
from pathlib import Path

import lightning as L
import pandas as pd
import torch

from src.data import transforms
from src.data.datamodule import MyDataModule
from src.models.lightning_module import RegressionModel
import src.models.models as models
from src.utils.config import load_config


def _make_transform(name):
	if name in (None, "None"):
		return None
	return getattr(transforms, name)()


def _build_data_slice(times, years_split):
	ts = pd.to_datetime(times)
	split = pd.Series("train", index=ts)

	_, val_years, test_years = years_split
	val_mask = (ts >= pd.Timestamp(val_years[0])) & (ts <= pd.Timestamp(val_years[1]))
	test_mask = (ts >= pd.Timestamp(test_years[0])) & (ts <= pd.Timestamp(test_years[1]))
	split.loc[val_mask] = "val"
	split.loc[test_mask] = "test"
	return split.to_numpy()


def main(config, checkpoint_path, output_path):
	transform_X = _make_transform(config["data"].get("transform_X"))
	transform_y = _make_transform(config["data"].get("transform_y"))

	datamodule = MyDataModule(
		data_in_name=config["data"]["data_in_name"],
		data_target_name=config["data"]["data_target_name"],
		data_path=config["data"]["data_path"],
		batch_size=config["data"]["batch_size"],
		transform_X=transform_X,
		transform_y=transform_y,
	)

	# Fit dataset-dependent transforms with the training split before prediction.
	datamodule.setup(stage="fit")

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
	if transform_y is not None and hasattr(transform_y, "inverse_transform"):
		target_inverse_transform = transform_y.inverse_transform

	lightning_model = RegressionModel.load_from_checkpoint(
		checkpoint_path,
		model=neural_network,
		learning_rate=config["trainer"]["learning_rate"],
		target_inverse_transform=target_inverse_transform,
		map_location="cpu",
	)

	trainer = L.Trainer(logger=False)
	prediction_batches = trainer.predict(lightning_model, datamodule=datamodule)

	preds = torch.cat([batch[0].detach().cpu() for batch in prediction_batches], dim=0)
	targets = torch.cat([batch[1].detach().cpu() for batch in prediction_batches], dim=0)
	time_indices = torch.cat([batch[2].detach().cpu() for batch in prediction_batches], dim=0).numpy()

	datamodule.setup(stage="predict")
	times = datamodule.dataset_full.times[time_indices]

	if target_inverse_transform is not None:
		preds = target_inverse_transform(preds)
		targets = target_inverse_transform(targets)

	years_split = (datamodule.train_years, datamodule.val_years, datamodule.test_years)

	df = pd.DataFrame(
		{
			"prediction": preds.squeeze(-1).numpy(),
			"target": targets.squeeze(-1).numpy(),
			"data_slice": _build_data_slice(times, years_split),
		},
		index=pd.to_datetime(times),
	)
	df.index.name = "time"
	df = df.sort_index()

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
		default="bccr-ml-course",
		help="Name of the output project folder where the run is stored.",
	)
	parser.add_argument(
		"--checkpoint-name",
		type=str,
		default="best.ckpt",
		help="Checkpoint filename under outputs/<project>/<run-id>/checkpoints/.",
	)
	parser.add_argument(
		"--checkpoint",
		type=str,
		default=None,
		help="Deprecated alias of --checkpoint-name.",
	)
	parser.add_argument(
		"--output",
		type=str,
		default=None,
		help="Path for the predictions to be saved.",    
	)
	args = parser.parse_args()
	if args.checkpoint:
		args.checkpoint_name = args.checkpoint

	config_path = f"outputs/{args.project_root}/{args.run_id}/config.yaml"
	checkpoint_path = f"outputs/{args.project_root}/{args.run_id}/checkpoints/{args.checkpoint_name}"
	run_output_dir = Path(f"outputs/{args.project_root}/{args.run_id}")
	if args.output is None:
		args.output = str(run_output_dir / "predictions.csv")
	config = load_config(config_path)
	main(config=config, checkpoint_path=checkpoint_path, output_path=args.output)