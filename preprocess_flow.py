"""
Precompute RAFT optical flow for every WAN latent clip.

For each *_wan.safetensors file, this script:
  1. Parses {start}-{length} from the filename to locate the pixel segment.
  2. Reads that segment from the matching mp4 under clips/.
  3. Subsamples T_latent frames evenly (matching WAN's 4x temporal compression).
  4. Runs RAFT between consecutive subsampled frames → T_latent - 1 flows.
  5. Saves to cache_dir/<stem>_flow.safetensors with key 'flow'.

Flow is stored at a reduced spatial resolution to save disk / memory.
The FlowEncoder uses AdaptiveAvgPool so the exact resolution doesn't matter.

Usage:
  python preprocess_flow.py
  python preprocess_flow.py --limit 10    # quick test
"""
import os
import re
import glob
import argparse
from typing import Optional

import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from safetensors.torch import save_file, load_file
from tqdm import tqdm


CLIPS_DIR = "/localdata/szhoubx/med_video/dataset/cholec80_action/clips"
CACHE_DIR = "/localdata/szhoubx/med_video/dataset/cholec80_action/cache_dir"

# Regex for filenames like "video01_clip02_00081-081_0832x0480_wan.safetensors"
FNAME_RE = re.compile(
    r"^(?P<clip>video\d+_clip\d+)_(?P<start>\d{5})-(?P<length>\d{3})_.*_wan\.safetensors$"
)

# Flow resolution (smaller is faster; FlowEncoder pools anyway)
FLOW_H, FLOW_W = 240, 416


@torch.no_grad()
def compute_flow_from_sampled(
    sampled: torch.Tensor,          # [T_latent, 3, H, W] uint8 (already subsampled)
    raft_model,
    raft_transforms,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Compute T_latent-1 flow fields between consecutive sampled frames."""
    if sampled.shape[0] < 2:
        return None

    sampled = F.interpolate(
        sampled.float(), size=(FLOW_H, FLOW_W),
        mode='bilinear', align_corners=False,
    ).to(torch.uint8)

    img1 = sampled[:-1].to(device)
    img2 = sampled[1:].to(device)
    img1_n, img2_n = raft_transforms(img1, img2)

    flows = raft_model(img1_n, img2_n)[-1]   # [T_latent-1, 2, FLOW_H, FLOW_W]
    return flows.to(torch.float16).cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None,
                        help="Process only the first N files (for testing)")
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load RAFT
    print("Loading RAFT (this downloads weights on first run)...")
    weights = Raft_Large_Weights.DEFAULT
    raft_model = raft_large(weights=weights).to(device).eval()
    raft_transforms = weights.transforms()

    # Collect all wan files
    wan_files = sorted(glob.glob(os.path.join(CACHE_DIR, "*_wan.safetensors")))
    wan_files = [f for f in wan_files if "_wan_te" not in f and "_text" not in f.lower()]

    if args.limit:
        wan_files = wan_files[:args.limit]

    print(f"Processing {len(wan_files)} wan files\n")

    ok, skip, fail = 0, 0, 0
    for wan_path in tqdm(wan_files):
        stem = os.path.basename(wan_path).replace("_wan.safetensors", "")
        out_path = os.path.join(CACHE_DIR, f"{stem}_flow.safetensors")

        if os.path.exists(out_path) and not args.overwrite:
            skip += 1
            continue

        m = FNAME_RE.match(os.path.basename(wan_path))
        if m is None:
            fail += 1
            continue

        clip_name = m.group('clip')              # e.g. "video01_clip02"
        start = int(m.group('start'))            # pixel frame start
        length = int(m.group('length'))          # pixel segment length

        mp4_path = os.path.join(CLIPS_DIR, f"{clip_name}.mp4")
        if not os.path.exists(mp4_path):
            fail += 1
            continue

        # Determine T_latent from the wan tensor so we only decode what we need
        wan_data_tmp = load_file(wan_path)
        wan_tensor = wan_data_tmp[list(wan_data_tmp.keys())[0]]
        T_latent = wan_tensor.shape[1]

        # Read only the T_latent evenly-spaced frames we actually use from mp4
        try:
            vr = VideoReader(mp4_path, ctx=cpu(0))
        except Exception:
            fail += 1
            continue

        end = min(start + length, len(vr))
        pixel_indices = torch.linspace(start, end - 1, T_latent).round().long().tolist()
        try:
            sampled = torch.from_numpy(
                vr.get_batch(pixel_indices).asnumpy()
            ).permute(0, 3, 1, 2)  # [T_latent, 3, H, W] uint8
        except Exception:
            fail += 1
            continue

        flow = compute_flow_from_sampled(
            sampled, raft_model, raft_transforms, device,
        )
        if flow is None:
            fail += 1
            continue

        save_file({'flow': flow}, out_path)
        ok += 1

    print(f"\nDone. ok={ok}  skipped_existing={skip}  failed={fail}")


if __name__ == "__main__":
    main()
