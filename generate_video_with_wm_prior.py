import os
import sys
import glob
import subprocess
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from genie.action import LatentAction, REPR_ACT_ENC, REPR_ACT_DEC
from genie.world_model import WorldModel

def run_cmd(cmd, cwd=None):
    print(f"\n[+] Running command: {' '.join(cmd)}\n")
    if cwd: 
        print(f"    in directory: {cwd}")
    subprocess.check_call(cmd, cwd=cwd)

def main():
    # ==============================================================
    # 配置你的模型路径 (硬编码默认值，可随时修改)
    # ==============================================================
    # 核心路径
    genie_dir = "/localdata/szhoubx/med_video/open-genie"
    tuner_dir = "/localdata/szhoubx/med_video/musubi-tuner"
    
    # 外部 Diffusion 与 LoRA
    wan_ckpt_dir = os.path.join(tuner_dir, "Wan2.1-T2V-1.3B-Diffusers")
    wan_lora_path = os.path.join(tuner_dir, "output_cholec80/wan_lora_cholec80-000003.safetensors")
    
    # 内部 LAM 与 World Model
    lam_ckpt = os.path.join(genie_dir, "action_extractor_continuous.pt")
    # 这里记得用你最好的 Epoch，如果有 ep160 就用 ep160
    wm_ckpt = os.path.join(genie_dir, "checkpoints/world_model_rssm_ep400.pt")
    
    # 提示词
    prompt = "A laparoscopic surgeon performs tissue retraction in a laparoscopic nephrectomy field, retracting perirenal tissue to widen the corridor to the renal hilum, slow camera tilt, realistic surgical lighting.""A laparoscopic surgeon performs aspiration in a laparoscopic nephrectomy operative field, suctioning irrigation around the renal hilum to keep the view clear, close-up scope perspective, realistic clinical footage, 4K."
    
    # 融合系数 (0.0=全用WM, 1.0=全用Diffusion, 0.7为优先推荐)
    alpha = 1
    
    print("=" * 60)
    print("  🚀 Diffusion + World Model 联合推理管道")
    print("=" * 60)

    # 准备临时目录
    temp_dir = os.path.join(genie_dir, "inference_out")
    os.makedirs(temp_dir, exist_ok=True)
    draft_dir = os.path.join(temp_dir, "draft")
    os.makedirs(draft_dir, exist_ok=True)

    # ==============================================================
    # [Step 1] 使用 Diffusion (Wan2.1 + LoRA) 生成潜变量草稿 Z_diff
    # ==============================================================
    print("\n[Step 1] Diffusion (Wan2.1+LoRA) 生成动作草稿 ...")
    # 我们限制到 81 帧 (对应 T=21) 480x832
    cmd_generate = [
        "/localdata/szhoubx/miniconda3/envs/medvideo/bin/python", "wan_generate_video.py",
        "--fp8", # 必须加，否则可能OOM
        "--task", "t2v-1.3B",
        "--dit", os.path.join(wan_ckpt_dir, "diffusion_pytorch_model.safetensors"),
        "--vae", os.path.join(wan_ckpt_dir, "Wan2.1_VAE.pth"),
        "--t5", os.path.join(wan_ckpt_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
        "--lora_weight", wan_lora_path,
        "--prompt", prompt,
        "--video_size", "480", "832",
        "--video_length", "81",
        "--output_type", "latent",
        "--save_path", draft_dir,
        "--seed", "42"  # 固定 seed 方便重复实验
    ]
    # 清空旧草稿
    old_drafts = glob.glob(os.path.join(draft_dir, "*.safetensors"))
    for f in old_drafts: os.remove(f)
        
    run_cmd(cmd_generate, cwd=tuner_dir)

    # 获取刚生成的 safetensors
    draft_files = glob.glob(os.path.join(draft_dir, "*.safetensors"))
    if not draft_files:
        raise RuntimeError("Z_diff draft generation failed. No safetensors found.")
    draft_file = sorted(draft_files)[-1]
    print(f" -> 草稿已生成: {draft_file}")


    # ==============================================================
    # [Step 2] 载入 LAM 与 World Model，执行物理校验与潜空间融合
    # ==============================================================
    print("\n[Step 2] 通过 World Model (WM) 进行动力学修正 ...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 挂载 LAM
    lam_model = LatentAction(
        enc_desc=REPR_ACT_ENC, dec_desc=REPR_ACT_DEC,
        d_codebook=8, inp_channels=16, inp_shape=(30, 52), n_embd=128
    ).to(device)
    lam_model.load_state_dict(torch.load(lam_ckpt, map_location=device, weights_only=True))
    lam_model.eval()

    # 2. 挂载 WM
    wm_model = WorldModel(
        action_dim=8, deter_dim=1024, stoch_dim=32, embed_dim=1024, in_channels=16
    ).to(device)
    wm_model.load_state_dict(torch.load(wm_ckpt, map_location=device, weights_only=True))
    wm_model.eval()

    # 3. 加载刚才拿到的 Z_diff = [16, T, 60, 104]
    draft_data = load_file(draft_file)
    Z_diff = draft_data["latent"].to(device) # [16, 21, 60, 104]
    Z_diff = Z_diff.float().unsqueeze(0)     # [1, 16, 21, 60, 104]
    orig_shape = Z_diff.shape
    
    # 4. 双流路由：降采样至 [30, 52] 给 LAM，WM 吃原始高清
    Z_diff_low = F.interpolate(Z_diff, size=(orig_shape[2], 30, 52), mode='trilinear', align_corners=False)
    
    with torch.no_grad():
        # LAM 吃糊图取动作
        act_embeds, _, _ = lam_model(Z_diff_low)
        act_seq = act_embeds[:, :-1] # [1, T-1, 8]
        
        # WM 吃高清图和高清第一帧
        obs_seq = Z_diff.permute(0, 2, 1, 3, 4).contiguous() 
        init_obs = obs_seq[:, 0]  # 第一帧绝对可信 [1, 16, 60, 104]
        
        # WM 进行纯高分物理规律演化 -> [1, T-1, 16, 60, 104]
        Z_wm = wm_model.rollout(init_obs, act_seq)
        
    # 5. 高分辨率特征级融合
    Z_wm_permuted = Z_wm.permute(0, 2, 1, 3, 4).contiguous() # [1, 16, T-1, 60, 104]
    
    # 因为 WM 直接生成了 [60, 104]，不再需要任何 upsample 拉长插值了！
    # 直接和最开始原始的高清 Z_diff 第一帧之后的帧进行融合
    Z_wm_highres = Z_wm_permuted
    
    # Z_diff 仍为超清的 [1, 16, T, 60, 104]
    Z_final_up = torch.zeros_like(Z_diff)
    Z_final_up[:, :, 0] = Z_diff[:, :, 0] # 冻结原画质超高清首帧
    Z_final_up[:, :, 1:] = alpha * Z_diff[:, :, 1:] + (1 - alpha) * Z_wm_highres
    print(f" -> 高分辨率原尺寸潜空间融合完成 (alpha = {alpha}).")
    
    # 移出 batch 维度，保存新 safetensors -> [16, 21, 60, 104]
    Z_out = Z_final_up.squeeze(0).cpu()
    blended_file = os.path.join(temp_dir, f"Z_blended_alpha{alpha}.safetensors")
    
    T_vae = Z_out.shape[1]
    # 保留给 Decode的元数据
    meta = {
        "height": "480", "width": "832", 
        "video_length": str((T_vae - 1)*4 + 1) # 推算原目标视频长度
    }
    save_file({"latent": Z_out}, blended_file, metadata=meta)
    print(f" -> 物理修正完毕并保存为: {blended_file}")

    # ==============================================================
    # [极限清理显存] 在拉起解码子进程前，必须清空主进程占用的所有深度学习变量
    # ==============================================================
    del lam_model
    del wm_model
    del Z_diff
    del Z_diff_down
    del obs_seq
    del init_obs
    del Z_wm
    del Z_wm_permuted
    del Z_wm_highres
    del Z_final_up
    del Z_out
    torch.cuda.empty_cache()

    # ==============================================================
    # [Step 3] 解码为 MP4
    # ==============================================================
    print("\n[Step 3] VAE 解码: Latent -> MP4 ...")
    final_mp4_dir = os.path.join(temp_dir, "final_mp4")
    cmd_decode = [
        "/localdata/szhoubx/miniconda3/envs/medvideo/bin/python", "wan_generate_video.py",
        "--fp8",
        "--task", "t2v-1.3B",
        "--dit", os.path.join(wan_ckpt_dir, "diffusion_pytorch_model.safetensors"),
        "--vae", os.path.join(wan_ckpt_dir, "Wan2.1_VAE.pth"),
        "--t5", os.path.join(wan_ckpt_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
        "--latent_path", blended_file,
        "--output_type", "video",
        "--save_path", final_mp4_dir
    ]
    run_cmd(cmd_decode, cwd=tuner_dir)
    print(f"\n✅ 渲染结束！最终的视频已保存在 {final_mp4_dir} 中，快去查看效果吧！")

if __name__ == "__main__":
    main()