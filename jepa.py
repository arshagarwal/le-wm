"""JEPA Implementation"""

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v

class JEPA(nn.Module):

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()

    def encode(self, info, patches=False):  # CLAW CHANGE: `patches` argument added
        """Encode observations and actions into embeddings.
        info: dict with pixels and action keys
        patches: also return the ViT patch tokens as info["patch_emb"], (B, T, num_patches, D). Off by default,
                 so existing callers get exactly the same outputs as before.
        """

        pixels = info['pixels'].float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...") # flatten for encoding
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]  # cls token
        emb = self.projector(pixels_emb)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)
        # ---- CLAW CHANGE START: optionally also return the ViT patch tokens (off by default) ----
        if patches:
            info["patch_emb"] = rearrange(output.last_hidden_state[:, 1:], "(b t) n d -> b t n d", b=b)
        # ---- CLAW CHANGE END ----

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def predict(self, emb, act_emb):
        """Predict next state embedding
        emb: (B, T, D)
        act_emb: (B, T, A_emb)
        """
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds

    ####################
    ## Inference only ##
    ####################

    def rollout(self, info, action_sequence, history_size: int = 3):
        """Rollout the model given an initial info dict and action sequence.
        pixels: (B, S, T, C, H, W)
        action_sequence: (B, S, T, action_dim)
         - S is the number of action plan samples
         - T is the time horizon
        """

        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        # copy and encode initial info dict
        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)
        _init = {k: detach_clone(v) for k, v in _init.items()}

        # flatten batch and sample dimensions for rollout
        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        # rollout predictor autoregressively for n_steps
        HS = history_size
        for t in range(n_steps):
            act_emb = self.action_encoder(act)
            emb_trunc = emb[:, -HS:]  # (BS, HS, D)
            act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
            pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
            emb = torch.cat([emb, pred_emb], dim=1)  # (BS, T+1, D)

            next_act = act_future[:, t : t + 1, :]  # (BS, 1, action_dim)
            act = torch.cat([act, next_act], dim=1)  # (BS, T+1, action_dim)

        # predict the last state
        act_emb = self.action_encoder(act)  # (BS, T, A_emb)
        emb_trunc = emb[:, -HS:]  # (BS, HS, D)
        act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
        pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
        emb = torch.cat([emb, pred_emb], dim=1)

        # unflatten batch and sample dimensions
        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout

        return info

    def criterion(self, info_dict: dict):
        """Compute the cost between predicted embeddings and goal embeddings."""
        pred_emb = info_dict["predicted_emb"]  # (B,S, T-1, dim)
        goal_emb = info_dict["goal_emb"]  # (B, S, T, dim)

        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)

        # return last-step cost per action candidate
        cost = F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, pred_emb.ndim)))  # (B, S)

        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """ Compute the cost of action candidates given an info dict with goal and initial state."""

        assert "goal" in info_dict, "goal not in info_dict"

        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]

        for k in info_dict:
            if k.startswith("goal_"):
                goal[k[len("goal_") :]] = goal.pop(k)

        goal.pop("action")
        goal = self.encode(goal)

        info_dict["goal_emb"] = goal["emb"]
        info_dict = self.rollout(info_dict, action_candidates)

        cost = self.criterion(info_dict)
        
        return cost


class _GradReverse(torch.autograd.Function):
    """Identity in the forward pass; multiplies the gradient by -scale in the backward pass."""

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.scale * grad, None


def grad_reverse(x, scale):
    return _GradReverse.apply(x, scale)


def adv_delta(progress, gamma=10.0, max_delta=0.999):
    """CLAW adversarial weight: min(2 / (1 + exp(-gamma * p)) - 1, max_delta), p = training progress in [0, 1]."""
    return min(2.0 / (1.0 + math.exp(-gamma * progress)) - 1.0, max_delta)


class _SelfBlock(nn.Module):
    """Pre-norm transformer block with (bidirectional) self-attention."""

    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim))

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class _CrossBlock(nn.Module):
    """V-JEPA 2's last probe block: the query token cross-attends to the tokens, then a residual MLP."""

    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim))

    def forward(self, q, x):
        kv = self.norm1(x)
        q = q + self.attn(q, kv, kv, need_weights=False)[0]
        return q + self.mlp(self.norm2(q))


class LAM(nn.Module):
    """Latent action model: an attentive probe (V-JEPA 2) over the encoder's patch tokens.

    forward(tok_t, tok_next) -> z: tokens (..., N, D) of two consecutive frames -> unit-norm latent action.
    The tokens of both frames form one sequence (plus a learned marker saying which frame each belongs to),
    go through depth-1 self-attention blocks, then a learnable query cross-attends to all of them (the last
    block), and a linear layer maps the query to latent_dim. cond(z) projects z to the vector the predictor
    takes in place of the real-action embedding.
    """

    def __init__(self, emb_dim=192, latent_dim=32, cond_dim=192, depth=4, heads=4, mlp_ratio=4.0):
        super().__init__()
        self.config = dict(emb_dim=emb_dim, latent_dim=latent_dim, cond_dim=cond_dim,
                           depth=depth, heads=heads, mlp_ratio=mlp_ratio)
        self.frame_embed = nn.Parameter(torch.randn(2, emb_dim) * 0.02)  # which frame a token comes from
        self.blocks = nn.ModuleList([_SelfBlock(emb_dim, heads, mlp_ratio) for _ in range(depth - 1)])
        self.query = nn.Parameter(torch.randn(1, 1, emb_dim) * 0.02)
        self.cross = _CrossBlock(emb_dim, heads, mlp_ratio)
        self.head = nn.Linear(emb_dim, latent_dim)
        self.proj = nn.Sequential(
            nn.Linear(latent_dim, cond_dim), nn.LayerNorm(cond_dim), nn.GELU(), nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, tok_t, tok_next):
        lead = tok_t.shape[:-2]
        x = torch.cat([tok_t + self.frame_embed[0], tok_next + self.frame_embed[1]], dim=-2)  # (..., 2N, D)
        x = x.reshape(-1, *x.shape[-2:])
        for blk in self.blocks:
            x = blk(x)
        q = self.cross(self.query.expand(len(x), -1, -1), x)[:, 0]
        return F.normalize(self.head(q), dim=-1).reshape(*lead, -1)

    def cond(self, z):
        return self.proj(z)
