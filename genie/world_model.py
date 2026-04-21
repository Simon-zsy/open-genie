import torch
import torch.nn as nn
from torch.distributions import kl_divergence, Normal
from genie.module.rssm import SpatialEncoder, SpatialDecoder, RSSMCell

class WorldModel(nn.Module):
    """
    DreamerV3 风格的潜空间世界模型 (World Model)。
    通过整合 VAE 特征和由 LAM 提取的动作序列，学习视频中的物理运动规律。
    """
    def __init__(self, action_dim=8, deter_dim=1024, stoch_dim=32, embed_dim=1024, hidden_dim=1024, in_channels=16):
        super().__init__()
        
        # 1. 核心模块初始化
        self.encoder = SpatialEncoder(in_channels=in_channels, embed_dim=embed_dim)
        self.decoder = SpatialDecoder(embed_dim=deter_dim + stoch_dim, out_channels=in_channels)
        self.rssm = RSSMCell(
            action_dim=action_dim, 
            deter_dim=deter_dim, 
            stoch_dim=stoch_dim, 
            embed_dim=embed_dim, 
            hidden_dim=hidden_dim
        )
        
    def forward(self, obs_seq, act_seq):
        """
        前向传播 (用于训练) - Teacher Forcing
        
        Args:
            obs_seq: [B, T, C, H, W] - 真实的 VAE 观测序列
            act_seq: [B, T-1, action_dim] - 由 LAM 抽取的连续动作序列
            
        Returns:
            recon_seq: [B, T, C, H, W] - 重建/预测的特征序列
            kl_loss: scalar - Prior 与 Posterior 间的正则化损失
        """
        B, T, C, H, W = obs_seq.shape
        
        # 1. 批量降维所有的观测特征 Z 
        obs_flat = obs_seq.reshape(B * T, C, H, W)
        embed_flat = self.encoder(obs_flat)
        embed_seq = embed_flat.reshape(B, T, -1)   # [B, T, embed_dim]
        
        # 2. 时序展开 RSSM
        states = []
        state = self.rssm.initial_state(B, obs_seq.device)
        
        # 第 0 帧处理: 没有上一时刻的动作，传入零填充作为初始化
        init_action = torch.zeros(B, act_seq.shape[-1], device=obs_seq.device)
        state = self.rssm.observe_step(state, init_action, embed_seq[:, 0])
        states.append(state)
        
        # 展开后续帧
        for t in range(1, T):
            action_t_minus_1 = act_seq[:, t-1] # 采取的前一个动作
            state = self.rssm.observe_step(state, action_t_minus_1, embed_seq[:, t])
            states.append(state)
            
        # 3. 解码状态进行重建预测 (Reconstruction)
        deter_states = torch.stack([s['deter'] for s in states], dim=1) # [B, T, deter_dim]
        stoch_states = torch.stack([s['stoch'] for s in states], dim=1) # [B, T, stoch_dim]
        
        dec_input = torch.cat([deter_states, stoch_states], dim=-1)
        dec_input_flat = dec_input.reshape(B * T, -1)
        
        recon_flat = self.decoder(dec_input_flat)
        recon_seq = recon_flat.reshape(B, T, C, H, W)
        
        # 4. 计算 KL 散度约束 (KL Divergence Loss)
        # 将 Prior 拉向 Posterior，以学习环境的转化规律
        prior_means = torch.stack([s['prior_mean'] for s in states], dim=1)
        prior_stds  = torch.stack([s['prior_std']  for s in states], dim=1)
        post_means  = torch.stack([s['post_mean']  for s in states], dim=1)
        post_stds   = torch.stack([s['post_std']   for s in states], dim=1)
        
        post_dist = Normal(post_means, post_stds)
        prior_dist = Normal(prior_means, prior_stds)
        kl = kl_divergence(post_dist, prior_dist)
        
        # Free nats 技术 (Dreamer经典技巧): 防止 Posterior 崩缩为 Prior 导致信息丢失
        # 我们强制保留最小 1.0 的 KL 正则信息界限
        kl_loss = torch.max(kl, torch.full_like(kl, 1.0)).mean()
        
        return recon_seq, kl_loss
        
    @torch.no_grad()
    def rollout(self, init_obs, act_seq):
        """
        开环推演 (Open-loop Imagination) - 用于推理或长程验证
        
        Args:
            init_obs: [B, C, H, W] - 第 t=0 帧的特征
            act_seq: [B, N, action_dim] - 未来 N 步连续动作
            
        Returns:
            imagined_obs_seq: [B, N, C, H, W] - 闭眼预测出的未来变化特征
        """
        B = init_obs.shape[0]
        N = act_seq.shape[1]
        
        # 编码第一帧
        init_embed = self.encoder(init_obs)
        
        # 初始化状态
        state = self.rssm.initial_state(B, init_obs.device)
        init_action = torch.zeros(B, act_seq.shape[-1], device=init_obs.device)
        state = self.rssm.observe_step(state, init_action, init_embed)
        
        imagined_states = []
        curr_state = state
        
        # 脱离观测，仅靠动作疯狂向未来推演 (Imagine)
        for t in range(N):
            action_t = act_seq[:, t]
            curr_state = self.rssm.imagine_step(curr_state, action_t)
            imagined_states.append(curr_state)
            
        # 集中解码生成的幻觉轨迹
        deter_states = torch.stack([s['deter'] for s in imagined_states], dim=1)
        stoch_states = torch.stack([s['stoch'] for s in imagined_states], dim=1)
        dec_input = torch.cat([deter_states, stoch_states], dim=-1)
        
        recon_flat = self.decoder(dec_input.reshape(B * N, -1))
        imagined_obs_seq = recon_flat.reshape(B, N, init_obs.shape[1], init_obs.shape[2], init_obs.shape[3])
        
        return imagined_obs_seq
