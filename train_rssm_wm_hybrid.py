"""
Train the RSSM world model using Hybrid Action Representation:
  clip-level CholecT50 triplet (semantic) + frame-level RAFT flow (motion).

Prerequisites:
  1. WAN latents cached at cache_dir/*_wan.safetensors
  2. Flow cached at  cache_dir/*_flow.safetensors  (run preprocess_flow.py first)
  3. Triplet labels in clips_captions.json

Single-GPU:
  python train_rssm_wm_hybrid.py

Multi-GPU (DDP):
  torchrun --nproc_per_node=N train_rssm_wm_hybrid.py
"""
import os
import json
import glob
import re
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from safetensors.torch import load_file
from tqdm import tqdm

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

from genie.world_model import WorldModel
from genie.hybrid_action import (
    HybridActionEncoder,
    encode_triplets,
    INST_TO_ID, VERB_TO_ID, TARG_TO_ID,
)


DATA_DIR        = "/localdata/szhoubx/med_video/dataset/cholec80_action"
CACHE_DIR       = os.path.join(DATA_DIR, "cache_dir")
CAPTIONS_JSON   = os.path.join(DATA_DIR, "clips_captions.json")

SEQ_LEN         = 16         # training temporal crop length
MAX_TRIPLETS    = 4          # clips with >4 triplets are truncated
ACTION_DIM      = 64
DIAG_EVERY_STEPS = 200       # run ablation diagnostics every N training steps

# Matches "video01_clip02_00081-081_0832x0480_wan.safetensors"
WAN_RE = re.compile(r"^(video\d+_clip\d+)_\d{5}-\d{3}_.*_wan\.safetensors$")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset: pairs (wan latent, triplet, flow) per wan shard
# ─────────────────────────────────────────────────────────────────────────────
class HybridDataset(Dataset):
    def __init__(self, cache_dir: str, captions_json: str):
        # Build clip_name → triplet list lookup
        with open(captions_json) as f:
            caps = json.load(f)
        self.triplet_lookup = {}
        for c in caps:
            video_name = os.path.basename(c['video']).replace('.mp4', '')
            self.triplet_lookup[video_name] = [tuple(t) for t in c.get('triplets', [])]

        # Enumerate wan shards that have a matching flow cache and valid triplets
        all_wan = sorted(glob.glob(os.path.join(cache_dir, "*_wan.safetensors")))
        self.items = []
        for wan_path in all_wan:
            if "_wan_te" in wan_path or "_text" in wan_path.lower():
                continue
            name = os.path.basename(wan_path)
            m = WAN_RE.match(name)
            if m is None:
                continue
            clip_name = m.group(1)
            if clip_name not in self.triplet_lookup:
                continue
            flow_path = wan_path.replace('_wan.safetensors', '_flow.safetensors')
            if not os.path.exists(flow_path):
                continue
            self.items.append((wan_path, flow_path, clip_name))

        print(f"HybridDataset: {len(self.items)} shards with matching wan + flow + triplet")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        wan_path, flow_path, clip_name = self.items[idx]

        wan_data = load_file(wan_path)
        latent = wan_data[list(wan_data.keys())[0]].float()      # [16, T, 60, 104]

        flow_data = load_file(flow_path)
        flow = flow_data['flow'].float()                          # [T-1, 2, 240, 416]

        triplets = self.triplet_lookup[clip_name]                 # list of (i, v, t)
        return latent, flow, triplets


def collate_hybrid(batch):
    """
    Randomly crop each item to SEQ_LEN latent frames (pad if shorter).
    Flow is cropped to the corresponding SEQ_LEN-1 frames.
    Triplets are padded to MAX_TRIPLETS.
    """
    latents_out, flows_out, triplet_lists = [], [], []
    for latent, flow, trips in batch:
        C, T, H, W = latent.shape
        if T >= SEQ_LEN:
            start = torch.randint(0, T - SEQ_LEN + 1, (1,)).item()
            latent_c = latent[:, start:start + SEQ_LEN]
            flow_c   = flow[start:start + SEQ_LEN - 1]
        else:
            # Pad short clips
            pad = SEQ_LEN - T
            latent_c = F.pad(latent, (0, 0, 0, 0, 0, pad))
            # flow has T-1 entries; pad to SEQ_LEN-1
            flow_pad = (SEQ_LEN - 1) - flow.shape[0]
            flow_c   = F.pad(flow, (0, 0, 0, 0, 0, 0, 0, flow_pad)) \
                       if flow_pad > 0 else flow[: SEQ_LEN - 1]
        latents_out.append(latent_c)
        flows_out.append(flow_c)
        triplet_lists.append(trips)

    latents = torch.stack(latents_out)                            # [B, 16, 16, 60, 104]
    flows   = torch.stack(flows_out)                              # [B, 15, 2, 240, 416]
    trip_ids, trip_mask = encode_triplets(triplet_lists, max_k=MAX_TRIPLETS)
    return latents, flows, trip_ids, trip_mask


# ─────────────────────────────────────────────────────────────────────────────
# Ablation diagnostics: are triplet / flow / action actually contributing?
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_diagnostics(action_enc, wm, latents, flows, trip_ids, trip_mask, device):
    """
    Returns a dict of contribution ratios. A healthy model has all > 0.05.
    """
    action_enc.eval(); wm.eval()
    try:
        obs_seq = latents.permute(0, 2, 1, 3, 4).contiguous()

        def loss_of(act_seq):
            recon, kl = wm(obs_seq, act_seq)
            return F.mse_loss(recon, obs_seq).item()

        # 1. Normal forward
        act_real = action_enc(trip_ids, trip_mask, flows)
        loss_real = loss_of(act_real)

        # 2. Zero out the whole action vector (should be much worse)
        loss_zero = loss_of(torch.zeros_like(act_real))

        # 3. Keep flow, zero triplet  (measures triplet contribution)
        act_flow_only = action_enc(
            torch.zeros_like(trip_ids), torch.zeros_like(trip_mask), flows,
        )
        loss_flow_only = loss_of(act_flow_only)

        # 4. Keep triplet, zero flow  (measures flow contribution)
        act_trip_only = action_enc(trip_ids, trip_mask, torch.zeros_like(flows))
        loss_trip_only = loss_of(act_trip_only)

        # Diversity of the action vector
        act_std_time  = act_real.std(dim=1).mean().item()     # variation across time
        act_std_batch = act_real.std(dim=0).mean().item()     # variation across batch

        def ratio(worse, base):
            return (worse - base) / (base + 1e-8)

        return {
            'loss_real':      loss_real,
            'action_contrib': ratio(loss_zero,      loss_real),
            'triplet_contrib':ratio(loss_flow_only, loss_real),
            'flow_contrib':   ratio(loss_trip_only, loss_real),
            'act_std_time':   act_std_time,
            'act_std_batch':  act_std_batch,
        }
    finally:
        action_enc.train(); wm.train()


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # DDP setup
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')
        is_main = rank == 0
    else:
        rank, world_size = 0, 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        is_main = True

    batch_size = 32
    epochs = 200
    lr = 1.5e-4

    dataset = HybridDataset(CACHE_DIR, CAPTIONS_JSON)
    if len(dataset) == 0:
        print("No data found — did you run preprocess_flow.py?")
        return

    sampler = (DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
               if world_size > 1 else None)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=collate_hybrid,
        num_workers=4,
        pin_memory=True,
    )

    # Models
    action_enc = HybridActionEncoder(
        triplet_dim=64, flow_dim=32, action_dim=ACTION_DIM,
    ).to(device)

    wm = WorldModel(
        action_dim=ACTION_DIM,      # 64 instead of 8
        deter_dim=1024,
        stoch_dim=32,
        embed_dim=1024,
        hidden_dim=1024,
        in_channels=16,
    ).to(device)

    if world_size > 1:
        action_enc = nn.parallel.DistributedDataParallel(
            action_enc, device_ids=[rank], output_device=rank,
        )
        wm = nn.parallel.DistributedDataParallel(
            wm, device_ids=[rank], output_device=rank, find_unused_parameters=True,
        )

    optimizer = torch.optim.AdamW(
        list(action_enc.parameters()) + list(wm.parameters()),
        lr=lr, weight_decay=1e-5,
    )

    if is_main:
        os.makedirs("runs", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)
        writer = SummaryWriter(log_dir="runs/hybrid_wm_training")
        print(f"Start training: {len(dataset)} shards, {len(loader)} iters/epoch, bs={batch_size}")
    else:
        writer = None

    action_enc.train()
    wm.train()

    global_step = 0
    for epoch in range(epochs):
        if sampler:
            sampler.set_epoch(epoch)

        tot_loss = tot_recon = tot_kl = 0.0
        n_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}") if is_main else loader
        for latents, flows, trip_ids, trip_mask in pbar:
            latents   = latents.to(device, non_blocking=True)           # [B, 16, 16, 60, 104]
            flows     = flows.to(device, non_blocking=True)             # [B, 15, 2, 240, 416]
            trip_ids  = trip_ids.to(device, non_blocking=True)
            trip_mask = trip_mask.to(device, non_blocking=True)

            # [B, T, 16, 60, 104] for WM
            obs_seq = latents.permute(0, 2, 1, 3, 4).contiguous()

            # Action: [B, T-1, ACTION_DIM]
            act_seq = action_enc(trip_ids, trip_mask, flows)

            # Mild input noise for drift robustness
            noisy_obs = obs_seq + torch.randn_like(obs_seq) * 0.05

            optimizer.zero_grad()
            recon_seq, kl_loss = wm(noisy_obs, act_seq)
            recon_loss = F.mse_loss(recon_seq, obs_seq)
            loss = recon_loss + 0.5 * kl_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(action_enc.parameters()) + list(wm.parameters()), 1.0,
            )
            optimizer.step()

            tot_loss  += loss.item()
            tot_recon += recon_loss.item()
            tot_kl    += kl_loss.item()
            n_batches += 1
            global_step += 1

            if is_main:
                writer.add_scalar('step/loss',  loss.item(),       global_step)
                writer.add_scalar('step/recon', recon_loss.item(), global_step)
                writer.add_scalar('step/kl',    kl_loss.item(),    global_step)
                if isinstance(pbar, tqdm):
                    pbar.set_postfix(
                        loss=f"{loss.item():.3f}",
                        recon=f"{recon_loss.item():.3f}",
                        kl=f"{kl_loss.item():.3f}",
                    )

                # Periodic ablation diagnostics: detects collapse early
                if global_step % DIAG_EVERY_STEPS == 0:
                    ae = action_enc.module if isinstance(action_enc, nn.parallel.DistributedDataParallel) else action_enc
                    wm_m = wm.module if isinstance(wm, nn.parallel.DistributedDataParallel) else wm
                    diag = run_diagnostics(ae, wm_m, latents, flows, trip_ids, trip_mask, device)
                    for k, v in diag.items():
                        writer.add_scalar(f'diag/{k}', v, global_step)
                    print(
                        f"[diag @ step {global_step}] "
                        f"act_contrib={diag['action_contrib']*100:+.1f}%  "
                        f"trip_contrib={diag['triplet_contrib']*100:+.1f}%  "
                        f"flow_contrib={diag['flow_contrib']*100:+.1f}%  "
                        f"act_std(t)={diag['act_std_time']:.3f}"
                    )
                    if global_step >= 2000 and diag['action_contrib'] < 0.02:
                        print(
                            "WARNING: action_contrib < 2% after 2k steps — "
                            "likely collapse. Consider stopping and debugging."
                        )

        avg_loss = tot_loss / max(n_batches, 1)
        if world_size > 1:
            t = torch.tensor(avg_loss, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            avg_loss = (t / world_size).item()

        if is_main:
            avg_recon = tot_recon / max(n_batches, 1)
            avg_kl    = tot_kl    / max(n_batches, 1)
            writer.add_scalar('epoch/loss',  avg_loss,  epoch + 1)
            writer.add_scalar('epoch/recon', avg_recon, epoch + 1)
            writer.add_scalar('epoch/kl',    avg_kl,    epoch + 1)
            print(f"Epoch {epoch+1} avg_loss={avg_loss:.4f} recon={avg_recon:.4f} kl={avg_kl:.4f}")

            if (epoch + 1) % 20 == 0 or epoch == epochs - 1:
                wm_state  = wm.module.state_dict()  if isinstance(wm, nn.parallel.DistributedDataParallel) else wm.state_dict()
                act_state = action_enc.module.state_dict() if isinstance(action_enc, nn.parallel.DistributedDataParallel) else action_enc.state_dict()
                torch.save({
                    'wm': wm_state,
                    'action_enc': act_state,
                    'epoch': epoch + 1,
                    'action_dim': ACTION_DIM,
                }, f"checkpoints/hybrid_wm_ep{epoch+1}.pt")
                print(f"Saved checkpoints/hybrid_wm_ep{epoch+1}.pt")

    if world_size > 1:
        dist.destroy_process_group()
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
