import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from safetensors.torch import load_file
from tqdm import tqdm

# 环境优化 (加速算子)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

from genie.action import LatentAction, REPR_ACT_ENC, REPR_ACT_DEC
from genie.world_model import WorldModel

class SafetensorsDataset(Dataset):
    def __init__(self, data_dir):
        pattern = os.path.join(data_dir, "**/*_wan.safetensors")
        all_files = glob.glob(pattern, recursive=True)
        
        self.files = []
        for f in all_files:
            if "_wan_te" in f or "_text" in f.lower():
                continue
            try:
                data = load_file(f)
                key = list(data.keys())[0]
                tensor = data[key]
                if tensor.ndim == 4 and tensor.shape[0] == 16 and tensor.shape[2] == 60 and tensor.shape[3] == 104:
                    self.files.append(f)
            except Exception:
                continue
        
        print(f"WorldModel: Found {len(self.files)} valid VAE embedding files.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = load_file(self.files[idx])
        key = list(data.keys())[0]
        latent_tensor = data[key].float() # [16, T, 60, 104]
        return latent_tensor

def collate_fn_16_frames(batch):
    """
    World Model 截断对齐: 统一截取 16 帧长，防止 RNN 产生长程误差累积爆炸
    """
    seq_len = 16
    cropped_items = []
    for item in batch:
        c, t, h, w = item.shape
        if t > seq_len:
            # 随机裁剪 16 帧
            start = torch.randint(0, t - seq_len + 1, (1,)).item()
            item = item[:, start:start+seq_len, :, :]
            cropped_items.append(item)
        elif t < seq_len:
            # 不足则在时间维度末尾 Pad
            pad_amt = seq_len - t
            item = F.pad(item, (0, 0, 0, 0, 0, pad_amt))
            cropped_items.append(item)
        else:
            cropped_items.append(item)
    return torch.stack(cropped_items)  # Shape: [B, 16, seq_len, 60, 104]

def main():
    # 1. Initialize DDP
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')
        is_main_process = rank == 0
    else:
        rank = 0
        world_size = 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        is_main_process = True
        
    data_dir = "/localdata/szhoubx/med_video/dataset/cholec80_action/cache_dir"
    batch_size = 28  # RSSM 比长程 Transformer 省显存得多，可以开大
    epochs = 400     # 约 500k steps
    learning_rate = 1.5e-4
    
    # 2. 数据流
    dataset = SafetensorsDataset(data_dir)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=sampler, 
        collate_fn=collate_fn_16_frames, 
        shuffle=(sampler is None)
    )

    # 3. 载入并冻结提取器 (Latent Action Model)
    lam = LatentAction(
        enc_desc=REPR_ACT_ENC, dec_desc=REPR_ACT_DEC,
        d_codebook=8, inp_channels=16, inp_shape=(30, 52), n_embd=128
    ).to(device)
    
    ckpt_path = "action_extractor_continuous.pt"
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    lam.load_state_dict(state_dict)
    
    lam.eval()
    for param in lam.parameters():
        param.requires_grad = False # 绝对不能更新 LAM

    if is_main_process:
        print(" -> Frozen Latent Action Model Loaded.")
        os.makedirs("runs", exist_ok=True)
        writer = SummaryWriter(log_dir="runs/world_model_training")
    else:
        writer = None
        
    # 4. 初始化世界模型 (Spatial-RSSM)
    # 因为空间维度扩大了 4 倍，为了防止严重的信息漏斗
    # 将 embed_dim, deter_dim, hidden_dim 同步扩容到 4096
    wm = WorldModel(
        action_dim=8, 
        deter_dim=4096, 
        stoch_dim=128,   # 随机状态的维度也适度扩容
        embed_dim=4096, 
        hidden_dim=4096, 
        in_channels=16
    ).to(device)
    
    if world_size > 1:
        wm = nn.parallel.DistributedDataParallel(wm, device_ids=[rank], output_device=rank)

    optimizer = torch.optim.AdamW(wm.parameters(), lr=learning_rate, weight_decay=1e-5)

    # 5. 训练循环
    if is_main_process:
        print(f"Starting World Model Training on {device}...")
        
    wm.train()
    for epoch in range(epochs):
        if sampler:
            sampler.set_epoch(epoch)
            
        total_loss, total_kl, total_recon = 0, 0, 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"WM Epoch {epoch+1}/{epochs}") if is_main_process else dataloader
        for batch_videos in pbar:
            batch_videos = batch_videos.to(device)
            
            # ==========================================
            # 双流路由 (Dual-path Routing) 验证
            # ==========================================
            orig_shape = batch_videos.shape
            
            # 路径 A: LAM 看糊图取动作 (保持 30x52 避免 OOM)
            batch_videos_low = F.interpolate(batch_videos, size=(orig_shape[2], 30, 52), mode='trilinear', align_corners=False)
            with torch.no_grad():
                (act_seq, _, _), _ = lam.encode(batch_videos_low)
                act_seq = act_seq[:, :-1]  # [B, T-1, 8]
                
            # 路径 B: WM 吃高清原图 [60, 104]
            # [B, T, 16, 60, 104]
            obs_seq = batch_videos.permute(0, 2, 1, 3, 4).contiguous()
            
            # c. 数据增强: 输入添加噪声 (标准差 σ=0.05) 模拟漂移鲁棒性
            noisy_obs_seq = obs_seq + torch.randn_like(obs_seq) * 0.05
            
            # d. 训练 WM: 获取重建结果与 KL 正则
            optimizer.zero_grad()
            
            # (处理 DDP 模块的前缀调用)
            wm_module = wm.module if isinstance(wm, nn.parallel.DistributedDataParallel) else wm
            recon_seq, kl_loss = wm_module(noisy_obs_seq, act_seq)
            
            # 计算动力学重建损失 MSE
            recon_loss = F.mse_loss(recon_seq, obs_seq) # 注意：Target 是纯净的无噪声观测
            
            # 损失加权 L_total = L_dyn + 0.5 * L_kl
            loss = recon_loss + 0.5 * kl_loss
            
            # 优化步
            loss.backward()
            torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0) # 防止 RNN 梯度爆炸
            optimizer.step()
            
            # 记录数据
            total_loss += loss.item()
            total_kl += kl_loss.item()
            total_recon += recon_loss.item()
            num_batches += 1
            
            if is_main_process:
                global_step = epoch * len(dataloader) + num_batches
                writer.add_scalar('Train_Step/Total_Loss', loss.item(), global_step)
                writer.add_scalar('Train_Step/Recon_Loss', recon_loss.item(), global_step)
                writer.add_scalar('Train_Step/KL_Loss', kl_loss.item(), global_step)
            
            if is_main_process and isinstance(pbar, tqdm):
                pbar.set_postfix(loss=loss.item(), recon=recon_loss.item(), kl=kl_loss.item())

        # 全局同步与打点
        avg_loss = total_loss / max(num_batches, 1)
        if world_size > 1:
            avg_loss_tensor = torch.tensor(avg_loss, device=device)
            dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = (avg_loss_tensor / world_size).item()
            
        if is_main_process:
            epoch_avg_recon = total_recon / max(num_batches, 1)
            epoch_avg_kl = total_kl / max(num_batches, 1)
            
            # 记录到 Tensorboard
            writer.add_scalar('Train_Epoch/Avg_Total_Loss', avg_loss, epoch + 1)
            writer.add_scalar('Train_Epoch/Avg_Recon_Loss', epoch_avg_recon, epoch + 1)
            writer.add_scalar('Train_Epoch/Avg_KL_Loss', epoch_avg_kl, epoch + 1)
            
            print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f} (Recon: {epoch_avg_recon:.4f}, KL: {epoch_avg_kl:.4f})")
            
            # 定期保存权重
            if (epoch + 1) % 20 == 0 or epoch == epochs - 1:
                os.makedirs("checkpoints", exist_ok=True)
                model_state = wm.module.state_dict() if isinstance(wm, nn.parallel.DistributedDataParallel) else wm.state_dict()
                torch.save(model_state, f"checkpoints/world_model_rssm_ep{epoch+1}.pt")
                print(f"Saved WM Checkpoint to checkpoints/world_model_rssm_ep{epoch+1}.pt")

    if world_size > 1:
        dist.destroy_process_group()
        
    if is_main_process and writer is not None:
        writer.close()

if __name__ == "__main__":
    main()