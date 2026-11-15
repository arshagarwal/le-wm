import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from jepa import LAM, adv_delta, grad_reverse
from module import SIGReg
from utils import get_column_normalizer, get_img_preprocessor, SaveCkptCallback, StepCkptCallback


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    lambd = cfg.loss.sigreg.weight

    # CLAW: real actions are not used. encode() only runs the action encoder if "action" is in its input.
    output = self.model.encode({"pixels": batch["pixels"]})

    emb = output["emb"]  # (B, T, D)

    # CLAW: latent action z_t from consecutive frame embeddings (emb_t, emb_{t+1}), fed to the predictor
    # in place of the real-action embedding. T = ctx_len + 1, so there are ctx_len latent actions.
    assert n_preds == 1, "latent actions assume num_preds == 1"
    z = self.lam(emb[:, :-1], emb[:, 1:])  # (B, T-1, latent_dim), unit norm
    ctx_act = self.lam.cond(z)[:, :ctx_len]  # (B, ctx_len, D)

    ctx_emb = emb[:, :ctx_len]

    tgt_emb = emb[:, n_preds:] # label
    pred_emb = self.model.predict(ctx_emb, ctx_act) # pred

    # CLAW adversary: predict the same target from z alone (context frames replaced by zeros).
    # The gradient reversal makes the LAM (and encoder) remove whatever lets z alone predict the next frame.
    trainer = getattr(self, "trainer", None)
    progress = trainer.global_step / max(trainer.estimated_stepping_batches, 1) if trainer is not None else 0.0
    delta = adv_delta(progress)
    adv_act = self.lam.cond(grad_reverse(z, delta))[:, :ctx_len]
    adv_pred = self.model.predict(torch.zeros_like(ctx_emb), adv_act)

    # LeWM loss + adversarial loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
    output["adv_loss"] = (adv_pred - tgt_emb.detach()).pow(2).mean()
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"] + output["adv_loss"]

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
    
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    # Resume support: seed the shuffle with the step stored in the checkpoint so each restart sees a
    # different order (the train/val split above keeps using cfg.seed alone, so it never changes).
    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id)
    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    resumed_step = 0
    if ckpt_path.exists():
        resumed_step = int(torch.load(ckpt_path, map_location="cpu", weights_only=False)["global_step"])
        print(f"resuming from step {resumed_step}: shuffle seed {cfg.seed + resumed_step}")
    shuffle_gen = torch.Generator().manual_seed(cfg.seed + resumed_step)

    train = torch.utils.data.DataLoader(train_set, **cfg.loader, shuffle=True, drop_last=True, generator=shuffle_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)
    lam = LAM(emb_dim=cfg.embed_dim, latent_dim=cfg.latent_dim, cond_dim=cfg.embed_dim)

    optimizers = {
        'model_opt': {
            "modules": 'model|lam',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        lam = lam,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name, cfg=cfg.model, epoch_interval=1,
    )

    callbacks = [object_dump_callback]
    if cfg.get("ckpt_every_n_steps", 0) > 0:
        callbacks.append(StepCkptCallback(ckpt_path, cfg.ckpt_every_n_steps))

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()
    return


if __name__ == "__main__":
    run()
