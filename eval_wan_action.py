import os
import glob
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from genie.action import LatentAction, REPR_ACT_ENC, REPR_ACT_DEC


def evaluate_file(model, video_latents_down, device):
    """
    Returns a dict of metrics for one video clip.

    Test 1 — Next-frame prediction MSE (what we now train for):
        Encoder encodes video[0..t], act[t] is used by decoder to predict
        video[t+1]. This is the real bottleneck: the action must carry the
        delta information or prediction quality collapses.

    Test 2 — Zero-action ablation:
        Same forward pass but with act replaced by zeros. The gap between
        Test-1 MSE and this MSE shows how much the action channel actually
        contributes. If the gap is near zero, actions are being ignored.

    Test 3 — Frame-diff vs action-norm correlation:
        If actions encode inter-frame dynamics, their L2 norm should track
        the magnitude of actual visual change between frames. We compute
        Pearson correlation between ||act[t]|| and ||video[t+1] - video[t]||.
    """
    with torch.no_grad():
        # ── Encode ──────────────────────────────────────────────────────────
        (act, _, enc_video), _ = model.encode(video_latents_down)
        # act:       [1, T, 8]
        # enc_video: [1, C, T, H, W]

        T = video_latents_down.shape[2]
        if T < 2:
            return None

        past_enc       = enc_video[:, :, :-1]            # [1, C, T-1, H, W]
        future_gt      = video_latents_down[:, :, 1:]    # [1, C, T-1, H, W]
        act_for_decode = act[:, :-1]                     # [1, T-1, 8]

        # ── Test 1: next-frame prediction with real actions ─────────────────
        recon_real = model.decode(past_enc, act_for_decode)
        mse_real   = F.mse_loss(recon_real, future_gt).item()

        # ── Test 2: zero-action ablation ────────────────────────────────────
        act_zero   = torch.zeros_like(act_for_decode)
        recon_zero = model.decode(past_enc, act_zero)
        mse_zero   = F.mse_loss(recon_zero, future_gt).item()

        action_contribution = (mse_zero - mse_real) / (mse_zero + 1e-8)

        # ── Test 3: correlation between action norm and frame-diff norm ──────
        # frame-diff magnitude per step: ||video[t+1] - video[t]||
        frame_diffs = (future_gt - video_latents_down[:, :, :-1])
        diff_norms  = frame_diffs.reshape(T - 1, -1).norm(dim=-1)       # [T-1]
        act_norms   = act_for_decode.squeeze(0).norm(dim=-1)             # [T-1]

        if T - 1 > 1:
            diff_norms_np = diff_norms.cpu().float()
            act_norms_np  = act_norms.cpu().float()
            # Pearson correlation
            d_mean = diff_norms_np - diff_norms_np.mean()
            a_mean = act_norms_np  - act_norms_np.mean()
            denom  = (d_mean.norm() * a_mean.norm()).clamp(min=1e-8)
            corr   = (d_mean * a_mean).sum() / denom
            correlation = corr.item()
        else:
            correlation = float('nan')

    return {
        'mse_real':             mse_real,
        'mse_zero':             mse_zero,
        'action_contribution':  action_contribution,
        'correlation':          correlation,
        'T':                    T,
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # ── Model ─────────────────────────────────────────────────────────────
    inp_channels = 16
    inp_shape    = (30, 52)
    d_codebook   = 8

    model = LatentAction(
        enc_desc=REPR_ACT_ENC,
        dec_desc=REPR_ACT_DEC,
        d_codebook=d_codebook,
        inp_channels=inp_channels,
        inp_shape=inp_shape,
        n_embd=128
    ).to(device)

    checkpoint_path = "action_extractor_continuous.pt"
    if not os.path.exists(checkpoint_path):
        print(f"Error: checkpoint not found at {checkpoint_path}")
        return

    state_dict = torch.load(checkpoint_path, map_location=device)
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded weights from {checkpoint_path}\n")

    # ── Data ──────────────────────────────────────────────────────────────
    data_dir = "/localdata/szhoubx/med_video/dataset/cholec80_action/cache_dir"
    pattern  = os.path.join(data_dir, "**/*_wan.safetensors")
    all_files = glob.glob(pattern, recursive=True)

    test_files = []
    for f in all_files:
        if "_wan_te" in f or "_text" in f.lower():
            continue
        try:
            data = load_file(f)
            key  = list(data.keys())[0]
            t    = data[key]
            if t.ndim == 4 and t.shape[0] == 16 and t.shape[2] == 60 and t.shape[3] == 104:
                test_files.append(f)
        except Exception:
            continue
        if len(test_files) >= 5:
            break

    if not test_files:
        print("No valid files found.")
        return

    print(f"Evaluating {len(test_files)} files\n")
    print("=" * 65)

    agg = {'mse_real': 0., 'mse_zero': 0., 'action_contribution': 0., 'correlation': 0.}
    valid_n = 0

    for i, f in enumerate(test_files):
        data = load_file(f)
        key  = list(data.keys())[0]
        video_latents = data[key].float().unsqueeze(0).to(device)  # [1,16,T,60,104]

        # Downsample to match training resolution
        orig_shape = video_latents.shape
        video_down = F.interpolate(
            video_latents,
            size=(orig_shape[2], 30, 52),
            mode='trilinear',
            align_corners=False
        )

        m = evaluate_file(model, video_down, device)
        if m is None:
            continue

        print(f"Sample {i+1} [{os.path.basename(f)}]  (T={m['T']} frames)")
        print(f"  Test 1 — Next-frame pred MSE  (real action): {m['mse_real']:.5f}")
        print(f"  Test 2 — Next-frame pred MSE  (zero action): {m['mse_zero']:.5f}")
        print(f"           Action contribution  (higher=better): {m['action_contribution']*100:.1f}%")
        print(f"  Test 3 — Action-norm vs frame-diff correlation: {m['correlation']:.3f}")
        print("-" * 65)

        for k in agg:
            agg[k] += m[k]
        valid_n += 1

    if valid_n == 0:
        return

    n = valid_n
    print(f"\n{'='*65}")
    print(f"Average over {n} clips:")
    print(f"  Next-frame MSE  (real action) : {agg['mse_real']/n:.5f}")
    print(f"  Next-frame MSE  (zero action) : {agg['mse_zero']/n:.5f}")
    print(f"  Action contribution           : {agg['action_contribution']/n*100:.1f}%")
    print(f"  Action-norm / frame-diff corr : {agg['correlation']/n:.3f}")
    print()
    print("Interpretation guide:")
    print("  Action contribution ~0%  → action channel is being ignored (bad)")
    print("  Action contribution >10% → action carries real transition info (good)")
    print("  Correlation > 0.3        → action norm tracks visual change (good)")
    print("  Correlation ≈ 0          → action norm is unrelated to dynamics (bad)")


if __name__ == "__main__":
    main()
