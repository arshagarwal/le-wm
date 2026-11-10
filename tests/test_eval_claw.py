"""Tests for eval_claw.py: lookup, retrieval metrics, latent rollout/cost and the latent CEM solver (pure torch)."""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from gymnasium.spaces import Box
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import eval_claw
from eval_claw import Lookup, LatentCEMSolver, collect_latents, latent_cost, retrieval_metrics
from jepa import JEPA
from module import ARPredictor, GroundedActionEncoder, LatentActionModel, MLP

D, Z, A, IMG = 16, 8, 10, 4


class _Enc(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(3 * IMG * IMG, D)

    def forward(self, x, interpolate_pos_encoding=True):
        return SimpleNamespace(last_hidden_state=self.lin(x.flatten(1))[:, None, :])


def _model():
    torch.manual_seed(0)
    m = JEPA(
        encoder=_Enc(),
        predictor=ARPredictor(num_frames=3, input_dim=D, hidden_dim=D, output_dim=D, depth=1, heads=2, mlp_dim=32, dim_head=8),
        action_encoder=GroundedActionEncoder(input_dim=A, latent_dim=Z),
        lam=LatentActionModel(emb_dim=D, latent_dim=Z, hidden_dim=32),
        action_proj=MLP(input_dim=Z, hidden_dim=32, output_dim=D), emb_dim=D,
    ).eval()
    # AdaLN-zero starts with conditioning switched off (output ignores z); switch it on so tests are meaningful
    for blk in m.predictor.transformer.layers:
        nn.init.normal_(blk.adaLN_modulation[-1].weight, std=0.5)
    return m


def _lookup(n=300, seed=0):
    g = torch.Generator().manual_seed(seed)
    return Lookup(F.normalize(torch.randn(n, Z, generator=g), dim=-1), torch.randn(n, A, generator=g))


# ---- lookup and metrics

def test_lookup_returns_action_of_nearest_z_and_keeps_shape():
    lk = _lookup()
    assert torch.equal(lk.retrieve(lk.z[[3, 50, 299]]), lk.actions[[3, 50, 299]])
    q = F.normalize(torch.randn(4, 5, Z), dim=-1)
    out = lk.retrieve(q, chunk=7)
    assert out.shape == (4, 5, A)
    assert torch.equal(out[1, 2], lk.actions[torch.cdist(q[1, 2][None], lk.z).argmin()])


def test_metrics_good_when_z_determines_action_and_bad_when_noise():
    g = torch.Generator().manual_seed(1)
    mix = torch.randn(4, A, generator=g)

    def make(n):  # z = normalize([u, c]) is a bijection of u; the action is linear in u
        u = torch.randn(n, 4, generator=g)
        return F.normalize(torch.cat([u, torch.full((n, 4), 2.0)], 1), dim=-1), u @ mix

    (zr, ar), (zq, aq) = make(3000), make(300)
    m = retrieval_metrics(Lookup(zr, ar), zq, aq)
    assert m["knn_rel"] < 0.35 and m["chance_mse"] > 1.5 * m["mean_mse"]

    noise = Lookup(F.normalize(torch.randn(2000, Z, generator=g), dim=-1), torch.randn(2000, A, generator=g))
    m = retrieval_metrics(noise, F.normalize(torch.randn(500, Z, generator=g), dim=-1), torch.randn(500, A, generator=g))
    assert m["knn_rel"] > 1.5 and 0.9 < m["ridge_rel"] < 1.2


def test_collect_latents_pairs_each_transition_with_its_own_action():
    batch = [{"pixels": torch.randn(2, 4, 3, IMG, IMG), "action": torch.randn(2, 4, A)}]
    batch[0]["action"][0, 0, 0] = float("nan")
    z, a = collect_latents(_model(), batch, ctx_len=3, device="cpu")
    assert z.shape == (6, Z) and torch.allclose(z.norm(dim=-1), torch.ones(6), atol=1e-5)
    assert not torch.isnan(a).any()
    assert torch.equal(a[3:], batch[0]["action"][1, :3])  # actions 0..2 of the window, not 1..3


# ---- latent rollout / cost

def _info(b=1, s=3):
    return {
        "pixels": torch.randn(b, s, 1, 3, IMG, IMG),
        "goal": torch.randn(b, s, 1, 3, IMG, IMG),
        "action": torch.randn(b, s, 1, A),
    }


@torch.no_grad()
def test_latent_cost_equals_the_real_action_cost_when_z_is_the_grounded_action():
    """action_encoder = (grounding encoder -> action_proj), so feeding g(a) as z must reproduce JEPA.get_cost(a)."""
    m = _model()
    info = _info()
    acts = torch.randn(1, 3, 4, A)
    real = m.get_cost(dict(info), acts)  # existing real-action path
    lat = latent_cost(m, info, m.ground_encoder(acts))
    assert real.shape == lat.shape == (1, 3)
    assert torch.allclose(real, lat, atol=1e-5), (real, lat)
    assert real.std() > 1e-3  # different action sequences give different costs, so the match is not trivial


@torch.no_grad()
def test_latent_cost_depends_on_z_and_supports_several_envs():
    m = _model()
    info = _info(b=2, s=4)
    z = F.normalize(torch.randn(2, 4, 5, Z), dim=-1)
    c = latent_cost(m, info, z)
    assert c.shape == (2, 4) and torch.isfinite(c).all()
    assert not torch.allclose(c, latent_cost(m, info, z.flip(-1)))


# ---- solver

class _Planted:
    """Stands in for latent_cost: squared distance of the z-sequence to a hidden target per env."""

    def __init__(self, target):
        self.target = target

    def __call__(self, model, infos, z):
        return (z - self.target[:, None]).pow(2).sum(dim=(-1, -2))


def test_cem_recovers_planted_sequence_and_returns_retrieved_real_actions(monkeypatch):
    g = torch.Generator().manual_seed(0)
    lk = _lookup()
    B, H = 2, 5
    pick = torch.randint(len(lk), (B, H), generator=g)
    monkeypatch.setattr(eval_claw, "latent_cost", _Planted(lk.z[pick]))

    solver = LatentCEMSolver(nn.Linear(1, 1), lk, batch_size=2, num_samples=200, n_steps=15, topk=20, device="cpu", seed=0)
    solver.configure(action_space=Box(-1, 1, shape=(B, 2)), n_envs=B, config=SimpleNamespace(horizon=H, receding_horizon=H, action_block=5))
    out = solver.solve({"pixels": torch.zeros(B, 1, 3, IMG, IMG)})

    assert out["actions"].shape == (B, H, A) and out["latent_actions"].shape == (B, H, Z)
    assert torch.allclose(out["latent_actions"].norm(dim=-1), torch.ones(B, H), atol=1e-5)
    assert (out["actions"] == lk.actions[pick]).all(-1).float().mean() >= 0.8
