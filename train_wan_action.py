import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from safetensors.torch import load_file
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from genie.action import LatentAction, REPR_ACT_ENC, REPR_ACT_DEC

class SafetensorsDataset(Dataset):
    def __init__(self, data_dir):
        # Find all .safetensors files that are video VAE embeddings (NOT text embeddings)
        # Only keep files with shape [16, T, 60, 104]
        pattern = os.path.join(data_dir, "**/*_wan.safetensors")
        all_files = glob.glob(pattern, recursive=True)
        
        # Strict filtering: exclude text embeddings and validate shape
        self.files = []
        for f in all_files:
            # Skip text embedding files explicitly
            if "_wan_te" in f or "_text" in f.lower():
                continue
            
            # Load and validate shape: must be (16, T, 60, 104) for VAE embeddings
            try:
                data = load_file(f)
                key = list(data.keys())[0]
                tensor = data[key]
                # VAE embedding should have 16 channels and spatial shape 60x104
                if tensor.ndim == 4 and tensor.shape[0] == 16 and tensor.shape[2] == 60 and tensor.shape[3] == 104:
                    self.files.append(f)
            except Exception:
                # Skip files that fail to load or have unexpected format
                continue
        
        print(f"Found {len(self.files)} valid VAE embedding files (from {len(all_files)} total)")
        if len(self.files) == 0:
            print("WARNING: No valid VAE embedding files found!")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        data = load_file(file_path)
        
        # Get the VAE embedding tensor
        key = list(data.keys())[0]
        latent_tensor = data[key] # Shape: [16, T, 60, 104] (C, T, H, W)
        
        # Validate shape
        assert latent_tensor.shape[0] == 16
        assert latent_tensor.shape[2] == 60 and latent_tensor.shape[3] == 104
        
        # Convert to float32 for training
        return latent_tensor.float()

def collate_fn(batch):
    # batch is list of tensors (C, T, H, W)
    # Different safetensors might have different temporal lengths (T)
    # We pad the temporal dimension to the max length in this batch
    max_t = max(item.shape[1] for item in batch)
    
    padded_items = []
    masks = []
    for item in batch:
        c, t, h, w = item.shape
        # F.pad expects padding sizes starting from last dimension:
        # (W_left, W_right, H_top, H_bottom, T_front, T_back)
        pad_amt = max_t - t
        padded_item = torch.nn.functional.pad(item, (0, 0, 0, 0, 0, pad_amt))
        padded_items.append(padded_item)
        
        # Create a temporal mask: True for valid frames, False for padded ones
        mask = torch.cat([
            torch.ones(t, dtype=torch.bool),
            torch.zeros(pad_amt, dtype=torch.bool)
        ])
        masks.append(mask)
        
    return torch.stack(padded_items), torch.stack(masks)

def main():
    # 1. Initialize DDP
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # DDP initialization
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
    
    # 2. Setup config
    data_dir = "/localdata/szhoubx/med_video/dataset/cholec80_action/cache_dir"
    batch_size = 8
    epochs = 30
    learning_rate = 1e-4
    
    if is_main_process:
        print(f"Training with {world_size} GPUs, rank={rank}, device={device}")
    
    # 3. Dataset and Dataloader
    dataset = SafetensorsDataset(data_dir)
    if len(dataset) == 0:
        if is_main_process:
            print("No data found, please check path.")
        return
    
    # Use DistributedSampler for DDP
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=42 + rank
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn
    )
    
    # Note: Wan Latent Shape is [C=16, T=20, H=60, W=104]
    inp_channels = 16
    inp_shape = (30, 52)  
    # The continuous action dimensions to extract per frame (Information Bottleneck)
    d_codebook = 8 

    # 4. Model Initialization
    # Use the pre-designed blueprints from Genie action
    # We must match inp_channels and inp_shape perfectly!
    model = LatentAction(
        enc_desc=REPR_ACT_ENC,
        dec_desc=REPR_ACT_DEC,
        d_codebook=d_codebook,        # 连续向量维度，限制在一个很小的值来构成信息瓶颈
        inp_channels=inp_channels,    # Wan2.1 VAE 的通道数
        inp_shape=inp_shape,          # Wan2.1 VAE 的空间尺寸
        n_embd=128                    # 降低中间维度避免显存爆炸，原版是256
    ).to(device)

    # Wrap model with DDP if using multiple GPUs
    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=True  # 允许某些参数在 forward 中未被使用
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if is_main_process:
        print(f"Starting training on {device}...")
        os.makedirs("runs", exist_ok=True)
        writer = SummaryWriter(log_dir="runs/lam_training")
    else:
        writer = None
    
    # 5. Training Loop
    model.train()
    global_step = 0
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        # Set epoch for sampler (important for shuffling)
        sampler.set_epoch(epoch)
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}") if is_main_process else dataloader
        for batch_videos, batch_masks in pbar:
            batch_videos = batch_videos.to(device)
            batch_masks = batch_masks.to(device)
            
            # Downsample video latents spatially by 2x to reduce memory usage completely
            # shape is (B, C, T, H, W)
            orig_shape = batch_videos.shape
            batch_videos = F.interpolate(batch_videos, size=(orig_shape[2], 30, 52), mode='trilinear', align_corners=False)
            
            optimizer.zero_grad()
            
            # Forward pass: extracts continuous action embeddings, reconstructs video, returns loss
            act_embeds, loss, (rec_loss, q_loss) = model(batch_videos, mask=batch_masks)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            global_step += 1
            
            if is_main_process:
                writer.add_scalar('Train_Step/Total_Loss', loss.item(), global_step)
                writer.add_scalar('Train_Step/Recon_Loss', rec_loss.item(), global_step)
                writer.add_scalar('Train_Step/Q_Loss', q_loss.item(), global_step)
            
            if is_main_process and isinstance(pbar, tqdm):
                pbar.set_postfix(loss=loss.item(), rec_loss=rec_loss.item())
        
        # Average loss across all processes
        avg_loss = total_loss / max(num_batches, 1)
        if world_size > 1:
            avg_loss_tensor = torch.tensor(avg_loss, device=device)
            dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = (avg_loss_tensor / world_size).item()
        
        if is_main_process:
            print(f"Epoch {epoch+1} completed. Avg Loss: {avg_loss:.4f}")
            writer.add_scalar('Train_Epoch/Avg_Total_Loss', avg_loss, epoch + 1)
            
            # 定期保存 checkpoint
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                os.makedirs("checkpoints", exist_ok=True)
                model_state = model.module.state_dict() if isinstance(model, nn.parallel.DistributedDataParallel) else model.state_dict()
                torch.save(model_state, f"checkpoints/lam_ep{epoch+1}.pt")
                print(f"Checkpoint saved: checkpoints/lam_ep{epoch+1}.pt")
        
    # Cleanup
    if is_main_process and writer is not None:
        writer.close()
    
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
