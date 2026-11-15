"""CLAW-style evaluation on PushT: plan with CEM over latent actions, map them to real actions by nearest neighbour.

build_lookup       reference windows -> (latent action z, real action) pairs
latent_cem         CEM over latent action sequences (rollout_latent + latent_cost)
latent_to_action   planned z -> real actions by L2 nearest neighbour in the lookup
play_in_env        run the real actions in the PushT env, save a video, return success
run                glue: encode start/goal, plan, map, execute (open-loop), report success rate
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path

import hydra
import imageio
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig
from sklearn import preprocessing
from stable_worldmodel.world.world import _extract_init_goal
from torchvision import tv_tensors

from eval import get_dataset, get_episodes_length, img_transform
from jepa import LAM
from utils import get_column_normalizer, get_img_preprocessor


def load_lam(policy: str, device="cpu"):
    """Load the LAM saved next to a JEPA checkpoint: '<run>/weights_epoch_N.pt' -> '<run>/lam_epoch_N.pt'."""
    path = Path(swm.data.utils.get_cache_dir(None, sub_folder="checkpoints")) / policy
    ck = torch.load(path.with_name(path.name.replace("weights_", "lam_")), map_location=device)
    lam = LAM(**ck["config"])
    lam.load_state_dict(ck["state_dict"])
    return lam.to(device).eval()


@torch.no_grad()
def build_lookup(model, lam, loader, device="cpu"):
    """Map latent actions to real actions: run reference windows through the encoder and LAM.

    Each batch holds `pixels` (B, T, C, H, W) and `action` (B, T, A). The latent action z_t comes from
    (emb_t, emb_{t+1}) and describes the transition t -> t+1, so it is paired with action[t] (t < T-1).
    Returns z_bank (N, latent_dim) and a_bank (N, A); actions stay in the normalised space of training.
    """
    model.eval()
    lam.eval()
    zs, acts = [], []
    for batch in loader:
        emb = model.encode({"pixels": batch["pixels"].to(device)})["emb"]  # (B, T, D)
        z = lam(emb[:, :-1], emb[:, 1:])  # (B, T-1, latent_dim)
        a = batch["action"][:, :-1].to(device)  # (B, T-1, A)
        z, a = z.reshape(-1, z.shape[-1]), a.reshape(-1, a.shape[-1])
        ok = ~torch.isnan(a).any(dim=-1)  # windows crossing an episode end carry NaN actions
        zs.append(z[ok].cpu())
        acts.append(a[ok].cpu())
    return torch.cat(zs), torch.cat(acts)


def load_windows(cfg, history, num_windows=4096, batch_size=64, seed=0):
    """Reference windows from the training data, preprocessed exactly like in train.py."""
    dataset = swm.data.load_dataset(
        cfg.eval.dataset_name + ".h5", transform=None, cache_dir=cfg.get("cache_dir"),
        num_steps=history + 1, frameskip=cfg.plan_config.action_block,
        keys_to_load=["pixels", "action"], keys_to_cache=["action"],
    )
    transforms = [
        get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.eval.img_size),
        get_column_normalizer(dataset, "action", "action"),
    ]
    dataset.transform = spt.data.transforms.Compose(*transforms)
    idx = np.random.default_rng(seed).choice(len(dataset), size=min(num_windows, len(dataset)), replace=False)
    return torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, np.sort(idx)), batch_size=batch_size)


@torch.no_grad()
def rollout_latent(model, lam, ctx_emb, z, history=None):
    """Roll the predictor forward with latent actions (JEPA.rollout only takes real actions).

    ctx_emb: (N, C, D) embeddings of the C context frames; z: (N, H, latent_dim) latent actions, H >= C.
    Frame i is paired with action i, as in training. Returns (N, C + H - C + 1, D): the context followed
    by H - C + 1 predicted frames.
    """
    history = history or model.predictor.pos_embedding.shape[1]  # frames the predictor attends over
    emb, C = ctx_emb, ctx_emb.shape[1]
    act = lam.cond(z)  # (N, H, D)
    for t in range(z.shape[1] - C + 1):
        pred = model.predict(emb[:, -history:], act[:, : C + t][:, -history:])[:, -1:]
        emb = torch.cat([emb, pred], dim=1)
    return emb


def latent_cost(pred_emb, goal_emb):
    """Squared L2 distance between the last predicted embedding and the goal embedding. (N, T, D), (N, D) -> (N,)"""
    return (pred_emb[:, -1] - goal_emb).pow(2).sum(-1)


@torch.no_grad()
def latent_cem(model, lam, ctx_emb, goal_emb, horizon, num_samples=300, n_steps=30, topk=30,
               var_scale=1.0, history=None, seed=1234):
    """CEM over latent action sequences on the unit sphere.

    ctx_emb: (B, C, D) context embeddings; goal_emb: (B, D). Candidates are Gaussian samples, projected
    back onto the unit sphere (as the LAM's z is); the elites refit the Gaussian each iteration.
    Returns the best sampled sequence of the last iteration, (B, horizon, latent_dim), and its cost (B,).
    """
    history = history or model.predictor.pos_embedding.shape[1]
    B, C, D = ctx_emb.shape
    L = lam.config["latent_dim"]
    gen = torch.Generator(device=ctx_emb.device).manual_seed(seed)
    mean = torch.zeros(B, horizon, L, device=ctx_emb.device)
    std = var_scale * torch.ones_like(mean)
    ctx = ctx_emb[:, None].expand(B, num_samples, C, D).reshape(B * num_samples, C, D)
    goal = goal_emb[:, None].expand(B, num_samples, D).reshape(B * num_samples, D)

    for _ in range(n_steps):
        eps = torch.randn(B, num_samples, horizon, L, generator=gen, device=ctx_emb.device)
        cand = torch.nn.functional.normalize(mean[:, None] + std[:, None] * eps, dim=-1)
        pred = rollout_latent(model, lam, ctx, cand.reshape(B * num_samples, horizon, L), history)
        cost = latent_cost(pred, goal).reshape(B, num_samples)
        elite_cost, idx = cost.topk(topk, dim=1, largest=False)
        elite = torch.gather(cand, 1, idx[:, :, None, None].expand(-1, -1, horizon, L))
        mean, std = elite.mean(1), elite.std(1)

    best = torch.gather(cand, 1, idx[:, :1, None, None].expand(-1, -1, horizon, L))[:, 0]
    return best, elite_cost[:, 0]


@torch.no_grad()
def latent_to_action(z, z_bank, a_bank):
    """Map latent actions to real actions by L2 nearest neighbour in the lookup (k = 1, as in CLAW).

    z: (..., latent_dim), e.g. the (B, horizon, latent_dim) sequences from latent_cem.
    z_bank (N, latent_dim) and a_bank (N, A) come from build_lookup. Returns (..., A), in the same
    normalised action space as a_bank.
    """
    flat = z.reshape(-1, z.shape[-1])
    idx = torch.cdist(flat, z_bank.to(flat)).argmin(dim=1)
    return a_bank.to(flat)[idx].reshape(*z.shape[:-1], a_bank.shape[-1])


@torch.no_grad()
def encode_frame(model, tf, frame, device):
    """One uint8 HxWx3 frame -> embedding (1, 1, D)."""
    px = tf(tv_tensors.Image(torch.from_numpy(frame).permute(2, 0, 1)))
    return model.encode({"pixels": px[None, None].to(device)})["emb"]


def reset_env(world, init, goal):
    """Put the PushT env at the dataset start state and set the goal state."""
    world.reset()
    env = world.envs.envs[0].unwrapped
    env._set_state(init["state"][0]); env._set_goal_state(goal["goal_state"][0])


def step_env(world, actions, scaler, max_steps):
    """Run planned real actions in the env, at most max_steps env steps, stopping at the first success.
    actions: (blocks, action_dim x action_block), normalised. Returns (frames, success, final_distance)."""
    frames, success = [], False
    for a in scaler.inverse_transform(actions.reshape(-1, 2).cpu().numpy()).astype(np.float32)[:max_steps]:
        _, reward, term, _, infos = world.envs.step(a[None])
        frames.append(np.array(infos["pixels"][0][-1]))  # copy: the env reuses its pixel buffer
        success = success or bool(term[0])
        if success:
            break
    return frames, success, -float(reward[0])


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    """python eval_claw.py --config-name=pusht policy=<run>/weights_epoch_N.pt eval.img_size=<train img_size>
    Optional: +device=cpu  +num_ref=4096 (reference windows for the lookup)."""
    device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    model = swm.wm.utils.load_pretrained(cfg.policy).to(device).eval()
    lam = load_lam(cfg.policy, device)
    history = model.predictor.pos_embedding.shape[1]
    z_bank, a_bank = build_lookup(model, lam, load_windows(cfg, history, cfg.get("num_ref", 4096)), device)  # lookup

    # Eval setup (bookkeeping only): one PushT env, the image transform for the model, the expert dataset
    # (start frames, start states, goals), a scaler to turn normalised actions back into raw env actions,
    # the episode ids and lengths we sample start points from, and the folder for videos and results.
    cfg.world.num_envs, cfg.world.max_episode_steps = 1, 2 * cfg.eval.eval_budget
    world, tf, dataset = swm.World(**cfg.world, image_shape=(224, 224)), img_transform(cfg), get_dataset(cfg, cfg.eval.dataset_name)
    scaler = preprocessing.StandardScaler().fit(dataset.get_col_data("action"))  # to un-normalise actions
    eps = np.unique(dataset.get_col_data("episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"))
    lens, rng = dict(zip(eps, get_episodes_length(dataset, eps))), np.random.default_rng(cfg.seed)
    out = Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
    out.mkdir(parents=True, exist_ok=True)

    results = []
    for k, ep in enumerate(rng.choice(eps, cfg.eval.num_eval, replace=False)):
        start = int(rng.integers(0, lens[ep] - cfg.eval.goal_offset_steps - 1))
        init, goal, _ = _extract_init_goal(dataset, [ep], [start], cfg.eval.goal_offset_steps)
        reset_env(world, init, goal)
        goal_emb = encode_frame(model, tf, goal["goal"][0], device)[:, -1]                    # encode the goal once
        frames, success, steps = [init["pixels"][0]], False, 0
        while not success and steps < cfg.eval.eval_budget:                                   # MPC loop (as in CLAW)
            ctx_emb = encode_frame(model, tf, frames[-1], device)                             # 1. encode the current real frame
            z, _ = latent_cem(model, lam, ctx_emb, goal_emb, cfg.plan_config.horizon, cfg.solver.num_samples,
                              cfg.solver.n_steps, cfg.solver.topk, cfg.solver.var_scale, history, cfg.seed + steps)  # 2. CEM
            actions = latent_to_action(z, z_bank, a_bank)[0, : cfg.plan_config.receding_horizon]  # 3. real actions
            new, success, dist = step_env(world, actions, scaler, cfg.eval.eval_budget - steps)   # 4. execute the first k blocks
            frames, steps = frames + new, steps + len(new)
        imageio.mimsave(out / f"episode_{k}.mp4", frames, fps=10, quality=8)
        results.append(success)
        print(f"episode {k}: success={success} env_steps={steps} final_distance={dist:.1f}")
    rate = 100.0 * np.mean(results)
    print(f"success_rate: {rate:.1f}%")
    (out / cfg.output.filename).open("a").write(f"\nsuccess_rate: {rate:.1f}% ({results})\n")


if __name__ == "__main__":
    run()






