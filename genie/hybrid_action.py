"""
Hybrid action representation for surgical world models.

Combines clip-level CholecT50 triplet (semantic) with frame-level optical
flow (motion) into a per-frame action vector.

  action_t = fuse( triplet_emb, flow_emb_t )

- triplet_emb:  constant within a clip, carries "what operation is happening"
- flow_emb_t:   per-frame, carries "what is moving right now"
"""
from typing import List, Tuple
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# CholecT50 vocabulary (from analysis of clips_captions.json, 5209 clips)
# Index 0 is reserved for padding / null; real labels start at 1.
# ─────────────────────────────────────────────────────────────────────────────
INSTRUMENTS: List[str] = [
    'bipolar', 'clipper', 'grasper', 'hook', 'irrigator', 'scissors',
]
VERBS: List[str] = [
    'aspirate', 'clip', 'coagulate', 'cut', 'dissect',
    'grasp', 'irrigate', 'pack', 'retract',
]
TARGETS: List[str] = [
    'abdominal-wall/cavity', 'adhesion', 'blood-vessel', 'cystic-artery',
    'cystic-duct', 'cystic-pedicle', 'cystic-plate', 'fluid', 'gallbladder',
    'gut', 'liver', 'omentum', 'peritoneum', 'specimen-bag',
]

INST_TO_ID = {name: i + 1 for i, name in enumerate(INSTRUMENTS)}
VERB_TO_ID = {name: i + 1 for i, name in enumerate(VERBS)}
TARG_TO_ID = {name: i + 1 for i, name in enumerate(TARGETS)}

N_INST = len(INSTRUMENTS) + 1   # 7 (incl. null)
N_VERB = len(VERBS) + 1         # 10
N_TARG = len(TARGETS) + 1       # 15


def encode_triplets(
    triplets: List[List[Tuple[str, str, str]]],
    max_k: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a batch of per-clip triplet lists into padded id tensors.

    Args:
        triplets: list of length B; each element is a list of (inst, verb, targ) tuples.
        max_k:    max number of triplets per clip (longer lists are truncated).

    Returns:
        ids:  [B, max_k, 3]  long, 0 = padding/null
        mask: [B, max_k]     bool, True where a real triplet is present
    """
    B = len(triplets)
    ids = torch.zeros(B, max_k, 3, dtype=torch.long)
    mask = torch.zeros(B, max_k, dtype=torch.bool)
    for b, clip_trips in enumerate(triplets):
        for k, (inst, verb, targ) in enumerate(clip_trips[:max_k]):
            ids[b, k, 0] = INST_TO_ID.get(inst, 0)
            ids[b, k, 1] = VERB_TO_ID.get(verb, 0)
            ids[b, k, 2] = TARG_TO_ID.get(targ, 0)
            mask[b, k] = True
    return ids, mask


# ─────────────────────────────────────────────────────────────────────────────
# Triplet encoder: clip-level semantic action
# ─────────────────────────────────────────────────────────────────────────────
class TripletEncoder(nn.Module):
    """
    Embeds (instrument, verb, target) ids into a semantic action vector.
    Handles multiple triplets per clip via mask-aware mean pooling.
    """

    def __init__(self, emb_dim: int = 32, out_dim: int = 64):
        super().__init__()
        self.inst_emb = nn.Embedding(N_INST, emb_dim, padding_idx=0)
        self.verb_emb = nn.Embedding(N_VERB, emb_dim, padding_idx=0)
        self.targ_emb = nn.Embedding(N_TARG, emb_dim, padding_idx=0)
        self.proj = nn.Sequential(
            nn.Linear(emb_dim * 3, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        ids:  [B, K, 3]   long
        mask: [B, K]      bool
        returns: [B, out_dim]
        """
        inst = self.inst_emb(ids[..., 0])              # [B, K, emb_dim]
        verb = self.verb_emb(ids[..., 1])
        targ = self.targ_emb(ids[..., 2])
        trip = self.proj(torch.cat([inst, verb, targ], dim=-1))  # [B, K, out_dim]

        # Mask-aware mean pool over K valid triplets
        m = mask.unsqueeze(-1).float()                 # [B, K, 1]
        summed = (trip * m).sum(dim=1)
        count = m.sum(dim=1).clamp(min=1.0)
        return summed / count                          # [B, out_dim]


# ─────────────────────────────────────────────────────────────────────────────
# Flow encoder: frame-level motion representation
# ─────────────────────────────────────────────────────────────────────────────
class FlowEncoder(nn.Module):
    """
    Compresses a per-frame optical flow map [2, H, W] into a compact vector.
    Uses AdaptiveAvgPool to stay resolution-agnostic.
    """

    def __init__(self, out_dim: int = 32, pool_shape: Tuple[int, int] = (4, 7)):
        super().__init__()
        ph, pw = pool_shape
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, 4, stride=2, padding=1),     # H/2, W/2
            nn.ELU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),    # H/4, W/4
            nn.ELU(inplace=True),
            nn.AdaptiveAvgPool2d((ph, pw)),
            nn.Flatten(),
            nn.Linear(64 * ph * pw, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        # flow: [N, 2, H, W]  →  [N, out_dim]
        return self.net(flow)


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid: triplet (clip) + flow (frame)  →  per-frame action vector
# ─────────────────────────────────────────────────────────────────────────────
class HybridActionEncoder(nn.Module):
    """
    Per-frame action representation fusing clip-level triplet semantics
    with frame-level optical flow motion.

    Inputs:
      triplet_ids:  [B, K, 3]          long
      triplet_mask: [B, K]             bool
      flows:        [B, T-1, 2, H, W]  float  (T-1 flows for T frames)

    Output:
      action_seq:   [B, T-1, action_dim]
    """

    def __init__(
        self,
        triplet_dim: int = 64,
        flow_dim: int = 32,
        action_dim: int = 64,
    ):
        super().__init__()
        self.triplet_encoder = TripletEncoder(emb_dim=32, out_dim=triplet_dim)
        self.flow_encoder    = FlowEncoder(out_dim=flow_dim)

        self.fuse = nn.Sequential(
            nn.Linear(triplet_dim + flow_dim, action_dim),
            nn.ELU(inplace=True),
            nn.Linear(action_dim, action_dim),
            nn.LayerNorm(action_dim),
        )
        self.action_dim = action_dim

    def forward(
        self,
        triplet_ids: torch.Tensor,
        triplet_mask: torch.Tensor,
        flows: torch.Tensor,
    ) -> torch.Tensor:
        B, Tm1, _, H, W = flows.shape

        # Semantic (constant across time)
        trip_vec = self.triplet_encoder(triplet_ids, triplet_mask)   # [B, triplet_dim]
        trip_seq = trip_vec.unsqueeze(1).expand(-1, Tm1, -1)         # [B, T-1, triplet_dim]

        # Motion (per-frame)
        flow_flat = flows.reshape(B * Tm1, 2, H, W)
        flow_vec  = self.flow_encoder(flow_flat)                     # [B*(T-1), flow_dim]
        flow_seq  = flow_vec.reshape(B, Tm1, -1)                     # [B, T-1, flow_dim]

        # Fuse
        return self.fuse(torch.cat([trip_seq, flow_seq], dim=-1))    # [B, T-1, action_dim]


if __name__ == "__main__":
    # Quick smoke test
    B, T, H, W = 2, 10, 60, 104

    # Example clip triplets (variable-length)
    example_triplets = [
        [('grasper', 'retract', 'gallbladder')],
        [('hook', 'dissect', 'cystic-duct'),
         ('grasper', 'retract', 'gallbladder')],
    ]
    ids, mask = encode_triplets(example_triplets, max_k=4)

    flows = torch.randn(B, T - 1, 2, H, W)

    enc = HybridActionEncoder(triplet_dim=64, flow_dim=32, action_dim=64)
    action_seq = enc(ids, mask, flows)

    print(f"triplet_ids  : {ids.shape}")
    print(f"triplet_mask : {mask.shape}")
    print(f"flows        : {flows.shape}")
    print(f"action_seq   : {action_seq.shape}  (expect [{B}, {T-1}, 64])")
