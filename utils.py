import numpy as np
import torch
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback

def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


class ZScoreNormalizer:
    """Picklable z-score normalizer — uses a class instead of a closure so it
    survives pickle when DataLoader workers are spawned (required by LanceDataset)."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return ((x - self.mean) / self.std).float()


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()
    return dt.transforms.WrapTorchTransform(ZScoreNormalizer(mean, std), source=source, target=target)

class SaveCkptCallback(Callback):
    """Callback to save model checkpoint after each epoch using save_pretrained."""

    def __init__(self, run_name, cfg, epoch_interval: int = 1):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._save(pl_module.model, trainer.current_epoch + 1)

            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._save(pl_module.model, trainer.current_epoch + 1)

    def _save(self, model, epoch):
        from stable_worldmodel.wm.utils import save_pretrained
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_epoch_{epoch}.pt',
        )

class StepCkptCallback(Callback):
    """Save a full Lightning checkpoint (weights, optimizer, scheduler, step) every N steps to a
    fixed path, which is the path train.py resumes from. Written to a temp file and renamed so a
    runtime killed mid-write cannot leave a corrupt checkpoint."""

    def __init__(self, path, every_n_steps: int):
        super().__init__()
        self.path = str(path)
        self.every_n_steps = every_n_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if self.every_n_steps > 0 and step > 0 and step % self.every_n_steps == 0:
            self._save(trainer)

    def on_train_epoch_end(self, trainer, pl_module):
        if self.every_n_steps > 0:
            self._save(trainer)

    def _save(self, trainer):
        import os
        tmp = self.path + ".tmp"
        trainer.save_checkpoint(tmp)  # all ranks must call this; Lightning writes on rank 0 only
        if trainer.is_global_zero:
            os.replace(tmp, self.path)
