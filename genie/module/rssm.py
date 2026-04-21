import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class SpatialEncoder(nn.Module):
    def __init__(self, in_channels=16, embed_dim=1024):
        super().__init__()
        # Input: [B, 16, 60, 104]
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1), # -> [B, 32, 30, 52]
            nn.ELU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),          # -> [B, 64, 15, 26]
            nn.ELU(inplace=True),
            nn.Flatten(),
            nn.Linear(64 * 15 * 26, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
    def forward(self, x):
        # x is [B, C, H, W]
        return self.net(x)

class SpatialDecoder(nn.Module):
    def __init__(self, embed_dim=1024, out_channels=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 64 * 15 * 26),
            nn.ELU(inplace=True)
        )
        # 精确的卷积上采样，还原到原生分辨率 [B, 16, 60, 104]
        self.net = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=(0, 0)), # -> [B, 32, 30, 52]
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1, output_padding=(0, 0)) # -> [B, 16, 60, 104]
        )
        
    def forward(self, x):
        # x is [B, embed_dim]
        x = self.fc(x)
        x = x.view(-1, 64, 15, 26)
        return self.net(x)

class RSSMCell(nn.Module):
    def __init__(self, action_dim=8, deter_dim=1024, stoch_dim=32, embed_dim=1024, hidden_dim=1024):
        super().__init__()
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        
        # 1. 确定性状态更新 (GRU)
        # 输入: t-1时刻的随机状态 stoch_{t-1} + 动作 a_{t-1}
        self.gru = nn.GRUCell(stoch_dim + action_dim, deter_dim)
        
        # 2. 先验分布预测 (Prior): p(z_t | h_t)
        self.prior_mlp = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim, 2 * stoch_dim)
        )
        
        # 3. 后验分布感知 (Posterior): q(z_t | h_t, e_t)
        self.post_mlp = nn.Sequential(
            nn.Linear(deter_dim + embed_dim, hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim, 2 * stoch_dim)
        )
        
    def initial_state(self, batch_size, device):
        return {
            'deter': torch.zeros(batch_size, self.deter_dim, device=device),
            'stoch': torch.zeros(batch_size, self.stoch_dim, device=device)
        }
        
    def observe_step(self, prev_state, action, embed):
        """闭环观测步: 利用真实观测修正状态 (Teacher Forcing)"""
        # GRU Update: h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
        gru_input = torch.cat([prev_state['stoch'], action], dim=-1)
        deter_state = self.gru(gru_input, prev_state['deter'])
        
        # Prior: p(z_t | h_t)
        prior_stats = self.prior_mlp(deter_state)
        prior_mean, prior_logstd = torch.chunk(prior_stats, 2, dim=-1)
        prior_std = F.softplus(prior_logstd) + 0.1 # 加0.1防止数值崩溃
        
        # Posterior: q(z_t | h_t, e_t)
        post_input = torch.cat([deter_state, embed], dim=-1)
        post_stats = self.post_mlp(post_input)
        post_mean, post_logstd = torch.chunk(post_stats, 2, dim=-1)
        post_std = F.softplus(post_logstd) + 0.1
        
        # Sample stochastic state via Reparameterization
        post_dist = Normal(post_mean, post_std)
        stoch_state = post_dist.rsample() 
        
        return {
            'deter': deter_state,
            'stoch': stoch_state,
            'prior_mean': prior_mean,
            'prior_std': prior_std,
            'post_mean': post_mean,
            'post_std': post_std
        }

    def imagine_step(self, prev_state, action):
        """开环想象步: 仅依据动作推演未来 (Rollout/Inference)"""
        gru_input = torch.cat([prev_state['stoch'], action], dim=-1)
        deter_state = self.gru(gru_input, prev_state['deter'])
        
        prior_stats = self.prior_mlp(deter_state)
        prior_mean, prior_logstd = torch.chunk(prior_stats, 2, dim=-1)
        prior_std = F.softplus(prior_logstd) + 0.1
        
        prior_dist = Normal(prior_mean, prior_std)
        stoch_state = prior_dist.rsample()
        
        return {
            'deter': deter_state,
            'stoch': stoch_state,
            'prior_mean': prior_mean,
            'prior_std': prior_std
        }
