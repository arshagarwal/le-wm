"""Unit tests for the CLAW latent-action pieces. Pure torch: no Hydra/Lightning/data needed."""
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jepa import JEPA
from module import (
    ARPredictor,
    GroundedActionEncoder,
    LatentActionModel,
    MLP,
    adv_delta,
    grad_reverse,
)

D, Z, T, B, A = 192, 32, 4, 5, 10


class _FakeEncoder(nn.Module):
    pass


def _predictor():
    return ARPredictor(
        num_frames=3, input_dim=D, hidden_dim=D, output_dim=D,
        depth=2, heads=4, mlp_dim=64, dim_head=16,
    )


def _claw_model():
    m = JEPA(
        encoder=_FakeEncoder(),
        predictor=_predictor(),
        action_encoder=GroundedActionEncoder(input_dim=A, latent_dim=Z),
        projector=None,
        pred_proj=None,
        lam=LatentActionModel(emb_dim=D, latent_dim=Z, hidden_dim=64),
        action_proj=MLP(input_dim=Z, hidden_dim=64, output_dim=D),
        emb_dim=D,
    )
    # AdaLN-zero starts with conditioning switched off, which makes every gradient into z exactly 0;
    # switch it on so the gradient tests below are not trivially "0 == 0"
    for blk in m.predictor.transformer.layers:
        nn.init.normal_(blk.adaLN_modulation[-1].weight, std=0.5)
    return m


def test_grad_reverse_flips_and_scales():
    x = torch.randn(4, 3, requires_grad=True)
    y = grad_reverse(x, 0.5)
    assert torch.equal(y, x)
    (y * torch.arange(3.0)).sum().backward()
    assert torch.allclose(x.grad, -0.5 * torch.arange(3.0).expand(4, 3))


@pytest.mark.parametrize("p,expected", [(0.0, 0.0), (0.1, 0.462), (0.3, 0.905), (0.5, 0.987)])
def test_adv_delta_schedule(p, expected):
    assert adv_delta(p) == pytest.approx(expected, abs=2e-3)


def test_adv_delta_caps_and_clips():
    assert adv_delta(1.0) == pytest.approx(0.999)
    assert adv_delta(5.0) == adv_delta(1.0)
    assert adv_delta(-1.0) == 0.0


def test_lam_shapes_and_unit_norm():
    lam = LatentActionModel(emb_dim=D, latent_dim=Z, hidden_dim=64)
    emb = torch.randn(B, T, D)
    z = lam(emb[:, :-1], emb[:, 1:])
    assert z.shape == (B, T - 1, Z)
    assert torch.allclose(z.norm(dim=-1), torch.ones(B, T - 1), atol=1e-5)


def test_grounded_encoder_shapes_and_unit_norm():
    g = GroundedActionEncoder(input_dim=A, latent_dim=Z)
    out = g(torch.randn(B, T, A))
    assert out.shape == (B, T, Z)
    assert torch.allclose(out.norm(dim=-1), torch.ones(B, T), atol=1e-5)


def test_latent_forward_shapes():
    m = _claw_model()
    out = m.latent_forward(torch.randn(B, T, D), torch.randn(B, T, A), ctx_len=3, delta=0.5)
    assert out["z"].shape == (B, 3, Z)
    assert out["act_emb"].shape == (B, 3, D)
    assert out["adv_pred"].shape == (B, 3, D)
    assert out["ground_pred"].shape == (B, 3, Z)


def test_adversary_gradient_into_lam_is_reversed_and_scaled():
    """The predictor/null path gets the true gradient; the LAM gets -delta times it."""
    torch.manual_seed(0)
    m = _claw_model().eval()  # eval: no dropout, so two passes are comparable
    emb = torch.randn(B, T, D)
    tgt = torch.randn(B, 3, D)

    def lam_grad(delta):
        m.zero_grad()
        z = m.infer_latent_actions(emb)[:, :3]
        loss = (m.predict_null(z, delta) - tgt).pow(2).mean()
        loss.backward()
        return torch.cat([p.grad.flatten() for p in m.lam.parameters()]), m.null_token.grad.clone()

    g_half, null_half = lam_grad(0.5)
    g_one, null_one = lam_grad(1.0)
    assert g_one.abs().sum() > 0 and null_one.abs().sum() > 0  # gradients actually flow
    assert torch.allclose(g_half, 0.5 * g_one, atol=1e-6)  # scaled by delta
    assert torch.allclose(null_half, null_one)  # predictor-side grads are not scaled

    # sign: compare with a plain (non-reversed) pass
    m.zero_grad()
    z = m.infer_latent_actions(emb)[:, :3]
    (m.predict(m.null_token.expand(B, 3, -1), m.action_proj(z)) - tgt).pow(2).mean().backward()
    g_plain = torch.cat([p.grad.flatten() for p in m.lam.parameters()])
    assert torch.allclose(g_one, -g_plain, atol=1e-6)


def test_lam_input_gradient_flows_to_embeddings_by_default():
    m = _claw_model()
    assert m.detach_lam_input is False
    emb = torch.randn(B, T, D, requires_grad=True)
    m.infer_latent_actions(emb).sum().backward()
    assert emb.grad is not None and emb.grad.abs().sum() > 0


def test_detach_lam_input_flag_blocks_gradient():
    m = _claw_model()
    m.detach_lam_input = True
    emb = torch.randn(B, T, D, requires_grad=True)
    m.infer_latent_actions(emb).sum().backward()
    assert emb.grad is None


def test_ground_loss_never_reaches_lam_or_predictor():
    m = _claw_model()
    out = m.latent_forward(torch.randn(B, T, D), torch.randn(B, T, A), ctx_len=3, delta=1.0)
    (out["ground_pred"] - out["z"].detach()).pow(2).sum(-1).mean().backward()
    assert all(p.grad is None for p in m.lam.parameters())
    assert all(p.grad is None for p in m.predictor.parameters())
    assert all(p.grad is not None for p in m.ground_encoder.parameters())


def test_action_encoder_chain_gives_conditioning_dim():
    """Inference path used by rollout/get_cost: real action -> z -> predictor conditioning."""
    m = _claw_model()
    assert m.action_encoder(torch.randn(B * 2, 6, A)).shape == (B * 2, 6, D)


def test_baseline_has_no_claw_params():
    from module import Embedder

    m = JEPA(
        encoder=_FakeEncoder(), predictor=_predictor(),
        action_encoder=Embedder(input_dim=A, emb_dim=D),
    )
    assert m.lam is None
    names = set(m.state_dict())
    assert not any(k.startswith(("lam", "ground_encoder", "action_proj", "null_token")) for k in names)
