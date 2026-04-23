"""
Visual + statistical sanity check for precomputed RAFT flows.

- Visualization: saves a PNG grid for each inspected clip, overlaying
  (video frame | flow colorized) side-by-side. You eyeball whether the
  flow tracks real motion (tool/tissue movement).

- Statistics: per-clip mean / max flow magnitude, fraction of pixels
  with significant motion. Bad signs: magnitude near zero everywhere
  (flow failed), or magnitude saturated (flow diverged).

Usage:
  python inspect_flow.py                 # default: 5 clips → outputs/flow_inspect/
  python inspect_flow.py --n 10 --out /tmp/flow
"""
import os
import re
import glob
import argparse

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from torchvision.utils import flow_to_image
from safetensors.torch import load_file
from decord import VideoReader, cpu


CACHE_DIR = "/localdata/szhoubx/med_video/dataset/cholec80_action/cache_dir"
CLIPS_DIR = "/localdata/szhoubx/med_video/dataset/cholec80_action/clips"
WAN_RE = re.compile(r"^(video\d+_clip\d+)_(\d{5})-(\d{3})_.*_wan\.safetensors$")


def inspect_one(flow_path: str, out_dir: str):
    """Visualize flow + return quality stats for one clip."""
    stem = os.path.basename(flow_path).replace("_flow.safetensors", "")
    wan_path = os.path.join(CACHE_DIR, f"{stem}_wan.safetensors")
    m = WAN_RE.match(os.path.basename(wan_path))
    if m is None:
        return None

    clip_name, start, length = m.group(1), int(m.group(2)), int(m.group(3))
    mp4 = os.path.join(CLIPS_DIR, f"{clip_name}.mp4")

    flow = load_file(flow_path)['flow'].float()              # [T-1, 2, H, W]
    T_minus_1, _, H, W = flow.shape
    T_latent = T_minus_1 + 1

    # Pull the matching pixel frames (same sampling as preprocess_flow)
    vr = VideoReader(mp4, ctx=cpu(0))
    end = min(start + length, len(vr))
    idxs = torch.linspace(start, end - 1, T_latent).round().long().tolist()
    sampled = torch.from_numpy(vr.get_batch(idxs).asnumpy()).permute(0, 3, 1, 2)  # [T, 3, H0, W0]
    # Resize to the flow's resolution for side-by-side display
    sampled = F.interpolate(sampled.float(), size=(H, W), mode='bilinear',
                            align_corners=False).to(torch.uint8)

    # Flow → RGB via standard colorization
    flow_rgb = flow_to_image(flow)                            # [T-1, 3, H, W] uint8

    # Build a grid: row = time step, columns = [frame_t, flow_rgb]
    rows = []
    for t in range(T_minus_1):
        rows.append(torch.stack([sampled[t], flow_rgb[t]], dim=0))  # [2, 3, H, W]
    grid = torch.cat(rows, dim=0)                              # [2*(T-1), 3, H, W]

    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, f"{stem}.png")
    vutils.save_image(grid.float() / 255.0, out_png, nrow=2, padding=2)

    # Stats
    mag = flow.pow(2).sum(dim=1).sqrt()                        # [T-1, H, W]
    return {
        'stem': stem,
        'T': T_latent,
        'mean_mag': mag.mean().item(),
        'max_mag': mag.max().item(),
        'frac_moving': (mag > 1.0).float().mean().item(),      # >1 px/frame is "moving"
        'out_png': out_png,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=5, help="Number of clips to inspect")
    parser.add_argument('--out', type=str, default="outputs/flow_inspect")
    args = parser.parse_args()

    flow_files = sorted(glob.glob(os.path.join(CACHE_DIR, "*_flow.safetensors")))
    if not flow_files:
        print("No flow files found — run preprocess_flow.py first.")
        return

    # Pick a diverse sample: first, last, and some in the middle
    N = min(args.n, len(flow_files))
    step = max(1, len(flow_files) // N)
    picks = flow_files[::step][:N]

    print(f"Inspecting {len(picks)} flow files → {args.out}\n")
    print(f"{'clip':<50} {'T':>4} {'mean':>8} {'max':>8} {'moving%':>8}")
    print("-" * 85)

    for fp in picks:
        r = inspect_one(fp, args.out)
        if r is None:
            continue
        print(f"{r['stem'][:48]:<50} {r['T']:>4} "
              f"{r['mean_mag']:>8.3f} {r['max_mag']:>8.2f} "
              f"{r['frac_moving']*100:>7.1f}%")

    print(f"\nVisualizations saved to {args.out}/")
    print("\nWhat to look for:")
    print("  - mean_mag  should be roughly 0.5 ~ 5 (surgical scenes have moderate motion)")
    print("    * near 0  → flow failed / video is truly static")
    print("    * >20     → likely diverged / noisy")
    print("  - moving%   typically 20 ~ 60% for dissect/retract clips")
    print("  - PNG grid  left column = frame, right column = colored flow")
    print("    * flow colors should align with where tools/tissue move")
    print("    * uniform color = global camera drift; localized = tool motion (good)")


if __name__ == "__main__":
    main()
