import os
import glob
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

# 导入你的模块
from genie.action import LatentAction, REPR_ACT_ENC, REPR_ACT_DEC
from genie.world_model import WorldModel

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ==========================
    # 1. 结构与参数定义
    # ==========================
    inp_channels = 16
    inp_shape = (30, 52)
    d_codebook = 8 
    
    # ==========================
    # 2. 加载冻结的 LAM (获取动作)
    # ==========================
    lam_model = LatentAction(
        enc_desc=REPR_ACT_ENC,
        dec_desc=REPR_ACT_DEC,
        d_codebook=d_codebook,
        inp_channels=inp_channels,
        inp_shape=inp_shape,
        n_embd=128
    ).to(device)
    
    # 改为你实际的 action 模型路径
    lam_ckpt = "action_extractor_continuous.pt"
    if os.path.exists(lam_ckpt):
        lam_model.load_state_dict(torch.load(lam_ckpt, map_location=device))
        lam_model.eval()
        print("[+] LAM Loaded.")
    else:
        print(f"[-] ERROR: Missing LAM checkpoint at {lam_ckpt}")
        return

    # ==========================
    # 3. 加载训练好的 World Model 
    # ==========================
    wm_model = WorldModel(
        action_dim=8, 
        deter_dim=1024, 
        stoch_dim=32, 
        embed_dim=1024, 
        in_channels=16
    ).to(device)
    
    # 找最新的 wm checkpoint (如果有存成 pt)
    wm_ckpt = "/localdata/szhoubx/med_video/open-genie/checkpoints/world_model_rssm_ep200.pt"  # 替换成你实际跑出的最好 checkpoint 的名字
    if not os.path.exists(wm_ckpt):
        # 兜底找一个可能命名的文件
        checkpoints = sorted(glob.glob("world_model_*.pt"))
        if checkpoints:
            wm_ckpt = checkpoints[-1]
            
    if os.path.exists(wm_ckpt):
        wm_model.load_state_dict(torch.load(wm_ckpt, map_location=device))
        wm_model.eval()
        print(f"[+] World Model Loaded from {wm_ckpt}.")
    else:
        print(f"[-] ERROR: Missing WM checkpoint, please specify correct path.")
        return

    # ==========================
    # 4. 准备测试数据并进行长程推演
    # ==========================
    data_dir = "/localdata/szhoubx/med_video/dataset/cholec80_action/cache_dir"
    safetensors_files = glob.glob(os.path.join(data_dir, "*.safetensors"))
    
    test_files = []
    for f in safetensors_files:
        try:
            data = load_file(f)
            key = list(data.keys())[0]
            tensor = data[key]
            # 同样严格过滤形状
            if tensor.ndim == 4 and tensor.shape[0] == 16 and tensor.shape[2] == 60 and tensor.shape[3] == 104:
                test_files.append(f)
        except Exception:
            continue
        if len(test_files) >= 5:  # 测 5 个视频即可
            break

    total_mse = 0.0
    
    with torch.no_grad():
        for i, f in enumerate(test_files):
            data = load_file(f)
            key = list(data.keys())[0]
            # 扩展 B 维 -> [1, 16, T, 60, 104]
            video_latents = data[key].float().unsqueeze(0).to(device) 
            
            # ==========================================
            # 双流路由 (Dual-path Routing)
            # ==========================================
            orig_shape = video_latents.shape
            
            # 路径 A: LAM 看糊图取动作 (保持 30x52 避免 OOM)
            video_down = F.interpolate(video_latents, size=(orig_shape[2], 30, 52), mode='trilinear', align_corners=False)
            
            # [Step A]: 利用 LAM 提取动作 
            act_embeds, _, _ = lam_model(video_down) 
            act_seq = act_embeds[:, :-1]  # [1, T-1, 8]
            
            # 路径 B: WM 吃高清原图 [60, 104]
            # [1, T, 16, 60, 104]
            obs_seq = video_latents.permute(0, 2, 1, 3, 4).contiguous()
            
            # [Step C]: Open-Loop 闭眼推演 (Rollout)
            # 只给第一帧 obs_seq[:, 0] 和所有动作 act_seq
            init_obs = obs_seq[:, 0]
            imagined_seq = wm_model.rollout(init_obs, act_seq)
            
            # 对比真实未来序列: 从第二帧开始也就是 obs_seq[:, 1:]
            ground_truth_future = obs_seq[:, 1:]
            
            # 计算长程累计误差 MSE
            mse = F.mse_loss(imagined_seq, ground_truth_future)
            total_mse += mse.item()
            
            # 计算单步 MSE 的误差发散情况 (为了观察长程推断是否随时间崩盘)
            step_mse = F.mse_loss(imagined_seq, ground_truth_future, reduction='none')
            # 沿着 B,C,H,W 求均值，留着维度 T
            step_mse = step_mse.mean(dim=(0, 2, 3, 4)).tolist()
            
            print(f"Sample {i+1} [{os.path.basename(f)}]:")
            print(f"  [>] 时长 (T): {obs_seq.shape[1]} 帧 (推演长度: {obs_seq.shape[1]-1} 步)")
            print(f"  [>] Imagination MSE: {mse.item():.5f}")
            print(f"  [>] 误差累积 (T=1, T=N//2, T=N): {step_mse[0]:.4f} -> {step_mse[len(step_mse)//2]:.4f} -> {step_mse[-1]:.4f}")
            print("-" * 50)
            
    print(f"\nAverage Imagination MSE: {total_mse / len(test_files):.5f}")

if __name__ == "__main__":
    main()