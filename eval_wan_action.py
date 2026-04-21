import os
import glob
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from genie.action import LatentAction, REPR_ACT_ENC, REPR_ACT_DEC

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. 初始化模型 (配置必须与训练时完全一致)
    inp_channels = 16
    inp_shape = (30, 52)
    d_codebook = 8 
    
    model = LatentAction(
        enc_desc=REPR_ACT_ENC,
        dec_desc=REPR_ACT_DEC,
        d_codebook=d_codebook,
        inp_channels=inp_channels,
        inp_shape=inp_shape,
        n_embd=128
    ).to(device)
    
    # 2. 加载训练好的权重
    checkpoint_path = "action_extractor_continuous.pt"
    if not os.path.exists(checkpoint_path):
        print(f"Error: 找不到权重文件 {checkpoint_path}")
        return
        
    print(f"Loading weights from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    # ハンドル DDP 的 'module.' 前缀 (如果在多卡训练下保存的)
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
    model.load_state_dict(state_dict)
    model.eval()  # 设置为评估模式，关闭 dropout 等
    
    # 3. 筛选并加载一部分你的 cache_dir 测试数据
    data_dir = "/localdata/szhoubx/med_video/dataset/cholec80_action/cache_dir"
    pattern = os.path.join(data_dir, "**/*_wan.safetensors")
    all_files = glob.glob(pattern, recursive=True)
    
    test_files = []
    for f in all_files:
        if "_wan_te" in f or "_text" in f.lower():
            continue
        try:
            data = load_file(f)
            key = list(data.keys())[0]
            tensor = data[key]
            # 严格筛选形状正确的 VAE embeddings
            if tensor.ndim == 4 and tensor.shape[0] == 16 and tensor.shape[2] == 60 and tensor.shape[3] == 104:
                test_files.append(f)
        except Exception:
            continue
            
        # 仅抽取前 5 个有效文件作为快速验证
        if len(test_files) >= 5:
            break
            
    if not test_files:
        print("No valid VAE embedding files found for testing.")
        return
        
    print(f"\nFound {len(test_files)} valid files for evaluation.\n")
    
    # 4. 前向预测与误差计算
    total_mse = 0.0
    
    with torch.no_grad():
        for i, f in enumerate(test_files):
            data = load_file(f)
            key = list(data.keys())[0]
            # 添加 Batch 维度: Shape 变成 [1, 16, T, 60, 104]
            video_latents = data[key].float().unsqueeze(0).to(device) 
            
            # 使用和训练时同样的降采样过程
            orig_shape = video_latents.shape
            video_latents_down = F.interpolate(video_latents, size=(orig_shape[2], 30, 52), mode='trilinear', align_corners=False)
            
            # 模型推理
            # Encode -> 获取连续动作表征 (act_embeds)
            # Decode -> 利用动作表征重建特征，并计算 loss
            act_embeds, loss, (rec_loss, q_loss) = model(video_latents_down)
            
            mse = rec_loss.item()
            total_mse += mse
            
            print(f"Sample {i+1} [{os.path.basename(f)}]:")
            print(f"  [>] 原输入 Shape : {orig_shape}")
            print(f"  [>] 降采样 Shape : {video_latents_down.shape}")
            print(f"  [>] 动作向量 Shape: {act_embeds.shape} (极度压缩后的 a_t)")
            print(f"  [>] 重建误差(MSE): {mse:.6f}")
            print("-" * 50)
            
    print(f"\n平均重建误差 (Average MSE): {total_mse / len(test_files):.6f}")

if __name__ == "__main__":
    main()