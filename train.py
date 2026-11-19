import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from omegaconf import OmegaConf, open_dict

from jepa import LAM, adv_delta, grad_reverse
from module import SIGReg
from utils import get_column_normalizer, get_img_preprocessor, SaveCkptCallback, StepCkptCallback


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    lambd = cfg.loss.sigreg.weight

    # CLAW: real actions are not used. The LAM reads the patch tokens of consecutive frames and gives the
    # latent action z_t for the step t -> t+1, which is fed to the predictor in place of the real-action
    # embedding. T = ctx_len + 1, so there are ctx_len latent actions.
    assert n_preds == 1, "latent actions assume num_preds == 1"
    output = self.model.encode({"pixels": batch["pixels"]}, patches=True)
    emb, tokens = output["emb"], output["patch_emb"]  # (B, T, D), (B, T, N, D)
    z = self.lam(tokens[:, :-1], tokens[:, 1:])  # (B, T-1, latent_dim), unit norm
    ctx_act = self.lam.cond(z)[:, :ctx_len]  # (B, ctx_len, D)

    ctx_emb = emb[:, :ctx_len]

    tgt_emb = emb[:, n_preds:] # label
    pred_emb = self.model.predict(ctx_emb, ctx_act) # pred

    # ---- ACTION vs PREDICTION SIMILARITY START (monitoring only) -------------------------------------
    # z is 32-d and pred_emb is 192-d, so compare the conditioning vector cond(z) (= ctx_act, 192-d) with
    # the predicted embedding, both L2-normalised: L1 distance (sum over dims) and cosine similarity.
    a, p = F.normalize(ctx_act.detach(), dim=-1), F.normalize(pred_emb.detach(), dim=-1)
    self.log_dict({f"{stage}/act_pred_l1": (a - p).abs().sum(-1).mean(), f"{stage}/act_pred_cos": (a * p).sum(-1).mean()}, on_step=True, sync_dist=True)
    # ---- ACTION vs PREDICTION SIMILARITY END ---------------------------------------------------------

    # ---- STATE PROBE START (monitoring only) --------------------------------------------------------
    # L1 error of reading the true state (agent xy, block xy, block angle) out of the PREDICTED embedding
    # ("pred") and, as a reference, out of the encoder's own embedding of the same frame ("true"). Inputs are
    # detached, so only the two linear probe heads are trained. Plot the two together (see TENSORBOARD below).
    state = torch.nan_to_num(batch["state"][:, n_preds:, :5])  # (B, ctx_len, 5): state of each target frame
    for name, x in (("pred", pred_emb), ("true", tgt_emb)):
        output[f"probe_{name}_loss"] = F.l1_loss(self.probe[name](x.detach()), state)
    # ---- STATE PROBE END ----------------------------------------------------------------------------

    # ADVERSARIAL LOSS DISABLED for this experiment (LAM-only run). Uncomment the block below and the
    # two adv_loss lines to bring it back.
    # # CLAW adversary: predict the same target from z alone (context frames replaced by zeros).
    # # The gradient reversal makes the LAM (and encoder) remove whatever lets z alone predict the next frame.
    # trainer = getattr(self, "trainer", None)
    # progress = trainer.global_step / max(trainer.estimated_stepping_batches, 1) if trainer is not None else 0.0
    # delta = adv_delta(progress)
    # adv_act = self.lam.cond(grad_reverse(z, delta))[:, :ctx_len]
    # adv_pred = self.model.predict(torch.zeros_like(ctx_emb), adv_act)

    # LeWM loss (+ adversarial loss when enabled)
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
    # output["adv_loss"] = (adv_pred - tgt_emb.detach()).pow(2).mean()
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]  # + output["adv_loss"]
    # STATE PROBE: add the probe losses so the probe heads get trained (their inputs are detached, so this
    # adds no gradient to anything else). NOTE: "loss" now includes these two small terms.
    output["loss"] = output["loss"] + output["probe_pred_loss"] + output["probe_true_loss"]

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
    # ---- STATE PROBE START: two linear heads, embedding -> 5 state values (see lejepa_forward) ----
    probe = nn.ModuleDict({k: nn.Linear(cfg.embed_dim, 5) for k in ("pred", "true")})
    # ---- STATE PROBE END ----

    optimizers = {
        'model_opt': {
            "modules": 'model|lam|probe',  # probe: STATE PROBE heads
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        lam = lam,
        probe = probe,  # STATE PROBE
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
    # ---- TENSORBOARD START: live charts. version=0 makes a resumed run continue the same curves. ----
    elif cfg.tensorboard.enabled:
        logger = TensorBoardLogger(
            save_dir=cfg.tensorboard.save_dir, name=cfg.output_model_name, version=0,
            flush_secs=cfg.tensorboard.flush_secs,
        )
        # draw the pred and true probe curves on ONE chart (TensorBoard tab "Custom Scalars")
        logger.experiment.add_custom_scalars({"state probe (L1)": {
            "train": ["Multiline", ["fit/probe_pred_loss", "fit/probe_true_loss"]],
            "val": ["Multiline", ["validate/probe_pred_loss_epoch", "validate/probe_true_loss_epoch"]],
        }})
    # ---- TENSORBOARD END ----

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
