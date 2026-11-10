"""CLAW-style evaluation of a LeWM+CLAW checkpoint (CEM over latent actions + nearest-neighbour retrieval).

Plans like CLAW (Algorithm 1 / A.5): CEM searches sequences of latent actions z with the world model,
the best sequence is mapped to real actions by nearest neighbour in a lookup set of (z, real action)
pairs built from annotated reference windows, and the real actions are executed in PushT. Real actions
never enter the model.

  python eval_claw.py --config-name=pusht policy=<run>/weights_epoch_N.pt eval.img_size=<train img_size>
  python eval_claw.py ... +mode=retrieval     # only check how well z recovers the action (no simulator)

Extra options (use `+key=value`): mode (eval|retrieval), device, num_ref, num_query, num_workers.
Reuses config/eval/pusht.yaml; `eval.img_size` must equal the image size the checkpoint was trained with.
"""

import os

os.environ["MUJOCO_GL"] = "egl"

import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from stable_worldmodel.solver import CEMSolver

from eval import get_dataset, get_episodes_length, img_transform
from utils import get_column_normalizer, get_img_preprocessor


###############################
## z -> real action mapping  ##
###############################


class Lookup:
    """Nearest-neighbour map from latent actions to real actions (CLAW's retrieval step)."""

    def __init__(self, z, actions):
        self.z, self.actions = z.float().cpu(), actions.float().cpu()  # (N, latent_dim), (N, action_dim)

    def __len__(self):
        return len(self.z)

    def retrieve(self, z, chunk=2048):
        """z: (..., latent_dim) -> real action of the nearest stored z (L2), shape (..., action_dim)."""
        flat = z.reshape(-1, self.z.shape[1]).float().cpu()
        idx = torch.cat([torch.cdist(flat[i : i + chunk], self.z).argmin(1) for i in range(0, len(flat), chunk)])
        return self.actions[idx].reshape(*z.shape[:-1], -1)


@torch.no_grad()
def collect_latents(model, loader, ctx_len, device):
    """Encoder + LAM over training-style windows -> (z, the real action block of each transition)."""
    zs, acts = [], []
    for batch in loader:
        emb = model.encode({"pixels": batch["pixels"].to(device)})["emb"]
        zs.append(model.infer_latent_actions(emb)[:, :ctx_len].flatten(0, 1).float().cpu())
        acts.append(torch.nan_to_num(batch["action"], 0.0)[:, :ctx_len].flatten(0, 1).float().cpu())
    return torch.cat(zs), torch.cat(acts)


def retrieval_metrics(lookup, z, a, seed=0):
    """MSE (z-scored action space) of recovering the true action block a from z. *_rel: 0 = exact, 1 = no better
    than predicting the mean action."""
    g = torch.Generator().manual_seed(seed)
    x = torch.cat([lookup.z, torch.ones(len(lookup), 1)], 1)  # ridge z -> action: a learned-mapping reference
    w = torch.linalg.solve(x.T @ x + 1e-3 * torch.eye(x.shape[1]), x.T @ lookup.actions)
    mse = lambda p: (p - a).pow(2).mean().item()
    m = {
        "knn_mse": mse(lookup.retrieve(z)),
        "ridge_mse": mse(torch.cat([z, torch.ones(len(z), 1)], 1) @ w),
        "mean_mse": mse(lookup.actions.mean(0, keepdim=True).expand_as(a)),
        "chance_mse": mse(lookup.actions[torch.randint(len(lookup), (len(a),), generator=g)]),
    }
    m["knn_rel"], m["ridge_rel"] = m["knn_mse"] / m["mean_mse"], m["ridge_mse"] / m["mean_mse"]
    return m


def load_windows(cfg, model):
    """Training-style windows (same preprocessing and train/val split as train.py)."""
    train_cfg = OmegaConf.load(Path(__file__).parent / "config/train/lewm.yaml")
    history = model.predictor.pos_embedding.shape[1]  # history_size the checkpoint was trained with
    ds = swm.data.load_dataset(
        cfg.eval.dataset_name + ".h5", transform=None, frameskip=cfg.plan_config.action_block, num_steps=history + 1,
        keys_to_load=["pixels", "action"], keys_to_cache=["action"],
    )
    ds.transform = spt.data.transforms.Compose(
        get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.eval.img_size),
        get_column_normalizer(ds, "action", "action"),
    )
    gen = torch.Generator().manual_seed(train_cfg.seed)
    train, val = spt.data.random_split(ds, lengths=[train_cfg.train_split, 1 - train_cfg.train_split], generator=gen)
    return train, val, history


def window_loader(split, n, cfg, seed):
    idx = torch.randperm(len(split), generator=torch.Generator().manual_seed(seed))[:n].tolist()
    return torch.utils.data.DataLoader(
        torch.utils.data.Subset(split, idx), batch_size=64, shuffle=False, num_workers=cfg.get("num_workers", 0)
    )


###############################
## planning in latent space  ##
###############################


@torch.no_grad()
def latent_cost(model, infos, z):
    """Goal cost of latent-action sequences. infos: tensors (B, S, ...) as expanded by the solver; z: (B, S, T, Z).

    Same rollout and cost as JEPA.get_cost / rollout (history_len 1), but the predictor is conditioned on
    action_proj(z) instead of on an embedded real action. Returns (B, S).
    """
    B, S, T = z.shape[:3]
    assert infos["pixels"].size(2) == 1, "latent planning supports plan_config.history_len == 1 (the default)"
    emb = model.encode({"pixels": infos["pixels"][:, 0]})["emb"]  # (B, 1, D)
    goal = model.encode({"pixels": infos["goal"][:, 0]})["emb"][:, -1]  # (B, D)
    emb = emb.unsqueeze(1).expand(B, S, *emb.shape[1:]).reshape(B * S, 1, -1)
    act = model.action_proj(z.reshape(B * S, T, -1))  # (BS, T, D)
    hs = model.predictor.pos_embedding.shape[1]
    for t in range(T):  # frame t+1 is predicted from the last hs frames and the latent actions that follow them
        pred = model.predict(emb[:, -hs:], act[:, : t + 1][:, -hs:])[:, -1:]
        emb = torch.cat([emb, pred], dim=1)
    return (emb[:, -1].reshape(B, S, -1) - goal[:, None]).pow(2).sum(-1)


class LatentCEMSolver(CEMSolver):
    """CEMSolver that searches latent actions (unit sphere) and returns the retrieved real actions."""

    def __init__(self, model, lookup, **kwargs):
        super().__init__(model, **kwargs)
        self.lookup, self.latent_dim = lookup, lookup.z.shape[1]

    @torch.inference_mode()
    def solve(self, info_dict, init_action=None):
        start = time.time()
        n_envs = len(next(iter(info_dict.values())))
        best_z = torch.zeros(n_envs, self.horizon, self.latent_dim)
        for s in range(0, n_envs, self.batch_size):
            e = min(s + self.batch_size, n_envs)
            bs = e - s
            infos = {
                k: v[s:e].to(device=self.device, dtype=self.dtype if v.is_floating_point() else None)
                .unsqueeze(1).expand(bs, self.num_samples, *v.shape[1:])
                for k, v in info_dict.items() if torch.is_tensor(v)
            }
            mean = torch.zeros(bs, self.horizon, self.latent_dim, device=self.device, dtype=self.dtype)
            std = self.var_scale * torch.ones_like(mean)
            rows = torch.arange(bs, device=self.device)
            for _ in range(self.n_steps):
                eps = torch.randn(bs, self.num_samples, *mean.shape[1:], generator=self.torch_gen, device=self.device, dtype=self.dtype)
                cand = F.normalize(mean[:, None] + std[:, None] * eps, dim=-1)  # the LAM emits unit-norm z
                if mean.norm(dim=-1).min() > 0:
                    cand[:, 0] = F.normalize(mean, dim=-1)  # keep the current mean in the pool
                cost = latent_cost(self.model, infos, cand)
                top = torch.topk(cost, k=self.topk, dim=1, largest=False).indices
                elites = cand[rows[:, None], top]
                mean, std = elites.mean(1), elites.std(1)
            best_z[s:e] = cand[rows, top[:, 0]].cpu()  # CLAW executes the best sequence of the last iteration
        print(f"Latent CEM solve time: {time.time() - start:.4f} seconds")
        return {"actions": self.lookup.retrieve(best_z), "latent_actions": best_z}


###############################
## evaluation (as eval.py)   ##
###############################


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    assert cfg.get("policy", "random") != "random", "pass policy=<checkpoint>"
    device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    model = swm.wm.utils.load_pretrained(cfg.policy).to(device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    assert getattr(model, "lam", None) is not None, "needs a CLAW checkpoint (train with model=lewm_claw)"

    # -- lookup set from reference (training) windows
    train_set, val_set, history = load_windows(cfg, model)
    lookup = Lookup(*collect_latents(model, window_loader(train_set, cfg.get("num_ref", 4096), cfg, cfg.seed), history, device))
    print(f"lookup set: {len(lookup)} entries")

    if cfg.get("mode", "eval") == "retrieval":  # offline check: does z recover the true action? (no simulator)
        zq, aq = collect_latents(model, window_loader(val_set, cfg.get("num_query", 1024), cfg, cfg.seed + 1), history, device)
        for k, v in retrieval_metrics(lookup, zq, aq).items():
            print(f"{k:>12}: {v:.4f}")
        print("knn_rel / ridge_rel: 0 = recovers the action exactly, 1 = no better than predicting the mean action")
        return

    assert cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget, \
        "Planning horizon must be smaller than or equal to eval_budget"

    # -- the rest mirrors eval.py, with the latent solver in place of CEMSolver
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))
    transform = {"pixels": img_transform(cfg), "goal": img_transform(cfg)}

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        processor.fit(col_data[~np.isnan(col_data).any(axis=1)])
        process[col] = processor
        if col != "action":
            process[f"goal_{col}"] = process[col]

    s = cfg.solver
    solver = LatentCEMSolver(
        model, lookup, batch_size=s.batch_size, num_samples=s.num_samples, var_scale=s.var_scale,
        n_steps=s.n_steps, topk=s.topk, device=device, seed=cfg.seed,
    )
    policy = swm.policy.WorldModelPolicy(
        solver=solver, config=swm.PlanConfig(**cfg.plan_config), process=process, transform=transform
    )
    results_path = Path(swm.data.utils.get_cache_dir(), cfg.policy).parent

    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    max_start_per_row = np.array([max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)])
    valid_indices = np.nonzero(dataset.get_col_data("step_idx") <= max_start_per_row)[0]
    print(len(valid_indices), "valid starting points found for evaluation.")

    g = np.random.default_rng(cfg.seed)
    chosen = np.sort(valid_indices[g.choice(len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False)])
    eval_episodes = dataset.get_row_data(chosen)[col_name]
    eval_start_idx = dataset.get_row_data(chosen)["step_idx"]

    world.set_policy(policy)
    results_path.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    metrics = world.evaluate(
        dataset=dataset, start_steps=eval_start_idx.tolist(), goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget, episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True), video=results_path,
    )
    print(metrics)

    results_file = results_path / cfg.output.filename
    with results_file.open("a") as f:
        f.write(f"\n==== CONFIG ====\n{OmegaConf.to_yaml(cfg)}\n==== RESULTS ====\nmetrics: {metrics}\n")
        f.write(f"evaluation_time: {time.time() - start_time} seconds\n")


if __name__ == "__main__":
    run()
