import torch
import torch.nn as nn

class ALN(nn.Module):
    """
    ALN: Attention Layer Network
    由 K 层多头自注意力组成的模块，用于增强 CNN backbone 的特征表示能力。
    输入输出形状保持一致：[B, C, H, W]
    """
    def __init__(self, embed_dim=1024, num_heads=8, k_layers=1):
        super().__init__()
        self.k_layers = k_layers
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            for _ in range(k_layers)
        ])
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(k_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W] -> reshape -> [B, HW, C] -> attention -> [B, HW, C] -> reshape回 [B, C, H, W]
        """
        B, C, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)  # [B, HW, C]
        for attn, norm in zip(self.attn_layers, self.norm_layers):
            residual = x
            x, _ = attn(x, x, x)  # Self-attention
            x = norm(x + residual)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return x
