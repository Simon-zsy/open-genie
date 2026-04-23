"""
Evaluate the trained Hybrid World Model checkpoint.

Three tests
-----------
1. Teacher-forced reconstruction
   WM sees ALL ground-truth frames via observe_step (upper-bound quality).
   Measures per-step MSE.

2. Open-loop rollout
   WM sees ONLY the first frame, then imagines the rest using imagine_step.
   Measures per-step MSE – this is the real generalization test.

3. Action ablation on rollout
   Repeat open-loop rollout with zero action vector.
   If rollout_mse << ablation_mse, action is genuinely conditioning the rollout.

Usage
-----
  python eval_world_model_hybrid.py --ckpt checkpoints/hybrid_wm_ep200.pt
  python eval_world_model_hybrid.py --ckpt checkpoints/hybrid_wm_ep200.pt --n 100
"""
import os
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Reuse dataset / collate from training
from train_rssm_wm_hybrid import HybridDataset, collate_hybrid, CACHE_DIR, CAPTIONS_JSON
from genie.world_model import WorldModel
from genie.hybrid_action import HybridActionEncoder


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    action_dim = ckpt.get('action_dim', 64)

    action_enc = HybridActionEncoder(triplet_dim=64, flow_dim=32, action_dim=action_dim).to(device)
    action_enc.load_state_dict(ckpt['action_enc'])
    action_enc.eval()

    wm = WorldModel(
        action_dim=action_dim,
        deter_dim=1024,
        stoch_dim=32,
        embed_dim=1024,
        hidden_dim=1024,
        in_channels=16,
    ).to(device)
    wm.load_state_dict(ckpt['wm'])
    wm.eval()

    print(f"Loaded checkpoint: epoch={ckpt.get('epoch','?')}  action_dim={action_dim}")
    return wm, action_enc


@torch.no_grad()
def teacher_forced_mse(wm, obs_seq, act_seq):
    """
    WM sees all ground-truth frames (observe_step at every t).
    Returns per-step MSE tensor: [T]
    """
    recon_seq, _ = wm(obs_seq, act_seq)          # [B, T, C, H, W]
    diff = (recon_seq - obs_seq) ** 2            # [B, T, C, H, W]
    return diff.mean(dim=(0, 2, 3, 4))           # [T]


@torch.no_grad()
def rollout_mse(wm, obs_seq, act_seq):
    """
    WM sees only obs_seq[:, 0], then imagines T-1 future frames.
    Returns per-step MSE tensor: [T-1]
    """
    init_obs = obs_seq[:, 0]                     # [B, C, H, W]
    imagined = wm.rollout(init_obs, act_seq)     # [B, T-1, C, H, W]
    gt = obs_seq[:, 1:]                          # [B, T-1, C, H, W]
    diff = (imagined - gt) ** 2
    return diff.mean(dim=(0, 2, 3, 4))           # [T-1]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help="Path to hybrid_wm_ep*.pt")
    parser.add_argument('--n',    type=int, default=50,   help="Number of clips to evaluate")
    parser.add_argument('--bs',   type=int, default=8,    help="Batch size")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    wm, action_enc = load_checkpoint(args.ckpt, device)

    dataset = HybridDataset(CACHE_DIR, CAPTIONS_JSON)
    loader  = DataLoader(
        dataset,
        batch_size=args.bs,
        shuffle=True,
        collate_fn=collate_hybrid,
        num_workers=4,
        pin_memory=True,
    )

    # ── Accumulate per-step MSE across N clips ──────────────────────────────
    tf_mse_acc   = None   # teacher-forced, [T]
    ro_mse_acc   = None   # open-loop rollout, [T-1]
    abl_mse_acc  = None   # ablation (zero action), [T-1]
    n_processed  = 0

    for latents, flows, trip_ids, trip_mask in loader:
        if n_processed >= args.n:
            break

        latents   = latents.to(device)
        flows     = flows.to(device)
        trip_ids  = trip_ids.to(device)
        trip_mask = trip_mask.to(device)

        obs_seq = latents.permute(0, 2, 1, 3, 4).contiguous()  # [B, T, 16, H, W]

        # Action: [B, T-1, action_dim]
        act_seq  = action_enc(trip_ids, trip_mask, flows)
        act_zero = torch.zeros_like(act_seq)

        tf_step   = teacher_forced_mse(wm, obs_seq, act_seq)
        ro_step   = rollout_mse(wm, obs_seq, act_seq)
        abl_step  = rollout_mse(wm, obs_seq, act_zero)

        if tf_mse_acc is None:
            tf_mse_acc  = tf_step.clone()
            ro_mse_acc  = ro_step.clone()
            abl_mse_acc = abl_step.clone()
            n_batches   = 1
        else:
            tf_mse_acc  += tf_step
            ro_mse_acc  += ro_step
            abl_mse_acc += abl_step
            n_batches   += 1

        n_processed += latents.shape[0]

    tf_mse_acc  /= n_batches
    ro_mse_acc  /= n_batches
    abl_mse_acc /= n_batches

    # ── Report ───────────────────────────────────────────────────────────────
    T    = tf_mse_acc.shape[0]
    Tm1  = ro_mse_acc.shape[0]

    print(f"\n{'='*60}")
    print(f"Evaluated {n_processed} clips  |  T={T} frames per clip")
    print(f"{'='*60}\n")

    # Test 1: teacher-forced
    print("Test 1 — Teacher-forced reconstruction (upper bound)")
    print(f"  Overall MSE : {tf_mse_acc.mean().item():.4f}")
    print(f"  Per step    : " +
          " ".join(f"t{t}={tf_mse_acc[t].item():.3f}" for t in range(T)))

    # Test 2: open-loop rollout
    print("\nTest 2 — Open-loop rollout (generalization)")
    print(f"  Overall MSE : {ro_mse_acc.mean().item():.4f}")
    print(f"  Per step    : " +
          " ".join(f"t{t}={ro_mse_acc[t].item():.3f}" for t in range(Tm1)))

    # Test 3: ablation
    abl_overall = abl_mse_acc.mean().item()
    ro_overall  = ro_mse_acc.mean().item()
    lift = (abl_overall - ro_overall) / (ro_overall + 1e-8) * 100
    print("\nTest 3 — Action ablation (rollout with zero action)")
    print(f"  Zero-action MSE : {abl_overall:.4f}")
    print(f"  Real-action MSE : {ro_overall:.4f}")
    print(f"  Action lift     : {lift:+.1f}%  "
          f"({'action helps' if lift > 5 else 'action weak — investigate'})")

    # ── Per-step drift summary ────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"{'step':>5} | {'teacher-forced':>14} | {'open-loop':>10} | {'zero-action':>11}")
    print(f"{'─'*60}")
    for t in range(Tm1):
        print(f"  t={t:<2} | {tf_mse_acc[t+1].item():>14.4f} | "
              f"{ro_mse_acc[t].item():>10.4f} | {abl_mse_acc[t].item():>11.4f}")
    print(f"{'─'*60}")
    print(f"  mean | {tf_mse_acc.mean().item():>14.4f} | "
          f"{ro_mse_acc.mean().item():>10.4f} | {abl_mse_acc.mean().item():>11.4f}")


if __name__ == "__main__":
    main()
