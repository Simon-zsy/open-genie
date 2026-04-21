import os
import glob
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from genie.action import LatentAction, REPR_ACT_ENC, REPR_ACT_DEC
from genie.world_model import WorldModel

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ==========================
    # 1. 路径与配置
    # ==========================
    cache_dir = "/localdata/szhoubx/med_video/dataset/cholec80_action/cache_dir"
    lam_ckpt = "/localdata/szhoubx/med_video/open-genie/action_extractor_continuous.pt"
    wm_ckpt = "/localdata/szhoubx/med_video/open-genie/checkpoints/world_model_rssm_ep400.pt"
    out_dir = "/localdata/szhoubx/med_video/open-genie/inference_out"
    os.makedirs(out_dir, exist_ok=True)

    # 找到一个缓存的 VAE 特征文件来做推演起点
    # 我们找一个 wan 结尾的 safetensors
    cached_files = glob.glob(os.path.join(cache_dir, "**/*_wan.safetensors"), recursive=True)
    if not cached_files:
        print("未找到缓存的数据文件！")
        return
    test_file = cached_files[0]
    print(f"\n[+] 选择测试片段: {test_file}")

    # ==========================
    # 2. 加载模型
    # ==========================
    print("[+] 正在加载 LAM 和 World Model...")
    lam_model = LatentAction(
        enc_desc=REPR_ACT_ENC, dec_desc=REPR_ACT_DEC,
        d_codebook=8, inp_channels=16, inp_shape=(30, 52), n_embd=128
    ).to(device)
    lam_model.load_state_dict(torch.load(lam_ckpt, map_location=device, weights_only=True))
    lam_model.eval()

    wm_model = WorldModel(
        action_dim=8, deter_dim=1024, stoch_dim=32, embed_dim=1024, in_channels=16
    ).to(device)
    wm_model.load_state_dict(torch.load(wm_ckpt, map_location=device, weights_only=True))
    wm_model.eval()

    # ==========================
    # 3. 处理输入数据
    # ==========================
    data = load_file(test_file)
    key = list(data.keys())[0]
    real_video_latent = data[key].float().unsqueeze(0).to(device) # [1, 16, T, 60, 104]
    
    orig_shape = real_video_latent.shape
    T = orig_shape[2]
    print(f"[>] 原始真实序列形状: {orig_shape} (T={T})")
    
    # 降频到 world model 尺寸
    video_down = F.interpolate(real_video_latent, size=(T, 30, 52), mode='trilinear', align_corners=False)
    
    # ==========================
    # 4. 提取动作指令 & 纯世界模型推演 (Pure Imagination)
    # ==========================
    with torch.no_grad():
        # a. 提取动作：给 LAM 看全片，提取出真实动作序列
        act_embeds, _, _ = lam_model(video_down) 
        act_seq = act_embeds[:, :-1] # [1, T-1, 8]
        
        # b. 切出第一帧作为唯一观测！
        obs_seq = video_down.permute(0, 2, 1, 3, 4).contiguous() # [1, T, 16, 30, 52]
        init_obs = obs_seq[:, 0]  # [1, 16, 30, 52]
        
        print("\n[+] 闭眼推演 (Open-Loop Rollout) 开始...")
        print(f"    输入: 第 0 帧 + 未来 {act_seq.shape[1]} 步动作，不看任何后续真实画面！")
        
        # c. 世界模型疯狂脑补未来！
        Z_wm = wm_model.rollout(init_obs, act_seq) # [1, T-1, 16, 30, 52]
        
        # 算一下和真实后续帧的 MSE，看看纯推演的漂移程度
        ground_truth_future = obs_seq[:, 1:]
        mse = F.mse_loss(Z_wm, ground_truth_future).item()
        print(f"    推演结束！未来全序列均方误差 (MSE) 为: {mse:.4f}")

    # ==========================
    # 5. 拼装推演结果并保存，以便拿去 VAE 解码可视化
    # ==========================
    # 我们把真实的第 0 帧，和推演出来的未来 T-1 帧拼起来，构成完整纯推演张量
    # 注意把时间维度 T 提回来 -> [1, 16, 1, 30, 52] 和 [1, 16, T-1, 30, 52]
    # 但由于之前的错误教训，这里我们把纯梦境放大回 [60, 104]:
    
    Z_wm_permuted = Z_wm.permute(0, 2, 1, 3, 4).contiguous() # [1, 16, T-1, 30, 52]
    Z_wm_highres = F.interpolate(Z_wm_permuted, size=(Z_wm.shape[1], 60, 104), mode='trilinear', align_corners=False)
    
    # 和最原始超高清的第 0 帧拼接起来
    pure_imagination_Z = torch.cat([real_video_latent[:, :, :1], Z_wm_highres], dim=2)
    
    # 拿掉 batch 维度，准备保存为 safetensors
    final_output = pure_imagination_Z.squeeze(0).cpu() # [16, T, 60, 104]
    
    out_file = os.path.join(out_dir, "pure_wm_imagination.safetensors")
    
    # metadata 为 Wan2.1 解码器提供参数
    meta = {
        "height": "480", "width": "832", 
        "video_length": str((T - 1)*4 + 1) 
    }
    save_file({"latent": final_output}, out_file, metadata=meta)
    
    print(f"\n✅ 已保存纯世界模型推演路线至: {out_file}\n")
    
    import subprocess
    
    # ==========================
    # 6. 清理显存并直接解码
    # ==========================
    print("[+] 正在清理显存防 OOM...")
    del lam_model
    del wm_model
    del real_video_latent
    del video_down
    del act_embeds
    del act_seq
    del obs_seq
    del init_obs
    del Z_wm
    del Z_wm_permuted
    del Z_wm_highres
    del pure_imagination_Z
    del final_output
    torch.cuda.empty_cache()

    print("[+] 正在直接调用 Wan VAE 解码潜变量为 mp4 视频...")
    tuner_dir = "/localdata/szhoubx/med_video/musubi-tuner"
    wan_ckpt_dir = os.path.join(tuner_dir, "Wan2.1-T2V-1.3B-Diffusers")
    
    cmd_decode = [
        "/localdata/szhoubx/miniconda3/envs/medvideo/bin/python", "src/musubi_tuner/wan_generate_video.py",
        "--fp8", "--fp8_t5", "--offload_inactive_dit", "--vae_cache_cpu",
        "--task", "t2v-1.3B",
        "--dit", os.path.join(wan_ckpt_dir, "diffusion_pytorch_model.safetensors"),
        "--vae", os.path.join(wan_ckpt_dir, "Wan2.1_VAE.pth"),
        "--t5", os.path.join(wan_ckpt_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
        "--latent_path", out_file,
        "--output_type", "video",
        "--save_path", out_dir
    ]
    
    print(f"    Running: {' '.join(cmd_decode)}")
    subprocess.check_call(cmd_decode, cwd=tuner_dir)

if __name__ == "__main__":
    main()