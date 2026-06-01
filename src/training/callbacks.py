import os
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint

class ValidationPlotsCallback(Callback):
    def on_fit_end(self, trainer, pl_module):
        y_pred_t = getattr(pl_module, "final_val_preds", None)
        y_true_t = getattr(pl_module, "final_val_targets", None)
        if y_pred_t is None or y_true_t is None:
            return

        if pl_module.logger is None or not hasattr(pl_module.logger, "experiment"):
            return

        try:
            import wandb

            y_pred = y_pred_t.view(-1).numpy()
            y_true = y_true_t.view(-1).numpy()
            residuals = y_pred - y_true

            # Scatter: predictions vs targets
            scatter_table = wandb.Table(
                data=[[float(t), float(p)] for t, p in zip(y_true, y_pred)],
                columns=["target", "prediction"],
            )
            scatter_plot = wandb.plot.scatter(
                scatter_table,
                "target",
                "prediction",
                title="Validation data: Prediction vs Target",
            )

            # Residual histogram
            residual_table = wandb.Table(
                data=[[float(r)] for r in residuals],
                columns=["residual"],
            )
            residual_hist = wandb.plot.histogram(
                residual_table,
                "residual",
                title="Validation data: Residuals",
            )

            # Time series (if val timestamps are available from the datamodule)
            logs = {
                "final_val_scatter": scatter_plot,
                "final_val_residual_hist": residual_hist,
            }

            datamodule = trainer.datamodule
            val_times = None
            if hasattr(datamodule, "dataset_val") and hasattr(datamodule.dataset_val, "times"):
                val_times = datamodule.dataset_val.times

            if val_times is not None:
                n = min(len(val_times), len(y_true), len(y_pred))
                ts_table = wandb.Table(
                    data=[
                        [str(val_times[i]), float(y_true[i]), float(y_pred[i])]
                        for i in range(n)
                    ],
                    columns=["time", "target", "prediction"],
                )
                ts_plot = wandb.plot.line_series(
                    xs=list(range(n)),
                    ys=[ts_table.get_column("target"), ts_table.get_column("prediction")],
                    keys=["target", "prediction"],
                    title="Validation data: Target vs Prediction over samples",
                    xname="val_index",
                )
                logs["final_val_timeseries"] = ts_plot

            pl_module.logger.experiment.log(logs)
        except Exception:
            pass

def get_checkpoint_callback(out_dir):
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(out_dir, "checkpoints"),
        filename="best",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
    )
    return checkpoint_callback

def get_early_stopping_callback():
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min"
    )
    return early_stopping_callback


def get_validation_plots_callback():
    return ValidationPlotsCallback()