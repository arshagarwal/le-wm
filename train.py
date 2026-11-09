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

from module import SIGReg, adv_delta
from utils import get_column_normalizer, get_img_preprocessor, SaveCkptCallback


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    # CLAW latent actions: z inferred from consecutive embeddings replaces the real-action conditioning
    latent = getattr(self.model, "lam", None) is not None
    if latent:
        assert n_preds == 1, "latent actions assume num_preds == 1"
        lcfg = cfg.loss.latent
        trainer = getattr(self, "trainer", None)
        progress = trainer.global_step / max(trainer.estimated_stepping_batches, 1) if trainer is not None else 0.0
        delta = adv_delta(progress, lcfg.gamma, lcfg.max_delta)
        lat = self.model.latent_forward(emb, batch["action"], ctx_len, delta)
        ctx_act = lat["act_emb"]

    tgt_emb = emb[:, n_preds:] # label
    pred_emb = self.model.predict(ctx_emb, ctx_act) # pred

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]  

    metrics = {}
    if latent:
        adv_tgt = tgt_emb.detach() if lcfg.detach_target else tgt_emb
        output["adv_loss"] = (lat["adv_pred"] - adv_tgt).pow(2).mean()
        # z -> real-action grounding; z is detached so no action label reaches the LAM or predictor
        output["ground_loss"] = (lat["ground_pred"] - lat["z"].detach()).pow(2).sum(-1).mean()
        output["loss"] = output["loss"] + output["adv_loss"] + lcfg.w_ground * output["ground_loss"]

        with torch.no_grad():
            z = lat["z"].float().flatten(0, 1)  # (B*ctx, latent_dim)
            sv = torch.linalg.svdvals(z - z.mean(0, keepdim=True))
            p = sv / sv.sum().clamp_min(1e-8)
            # adversary error relative to predicting the per-position batch mean (1.0 = no leakage)
            base = (tgt_emb - tgt_emb.mean(0, keepdim=True)).pow(2).mean().clamp_min(1e-8)
            metrics = {
                "delta": torch.tensor(delta),
                "z_std": z.std(0).mean(),
                "z_erank": torch.exp(-(p * p.clamp_min(1e-12).log()).sum()),
                "adv_rel": output["adv_loss"].detach() / base,
            }

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    losses_dict.update({f"{stage}/{k}": v for k, v in metrics.items()})
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

    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id)

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

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
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
