"""
CSCA_v4_cleaned: Channel-Spatial-Context Attention, Final Version

This file represents the final, cleaned, and fully functional implementation of the CSCA module,
incorporating all collaboratively designed enhancements and bug fixes.

Key Innovations:
- Coord2H: A dual-head coordinate attention mechanism that decouples horizontal and vertical
  pattern detection, uses axis-softmax with mean-rescaling for amplification, and aligns
  with the multi-head design of the CoT branch.
- Advanced CoT Attention: Features implicit multi-heads, temperature scaling, mean-rescaling,
  and hybrid gating for maximum expressiveness and control.
- ECA Mid-Attention: A lightweight and efficient channel attention for the middle-stage block.
- Stability Enhancements: Includes residual scaling, full affine normalization, and robust
  parameter registration (buffers).

Author: Based on user requirements and deep collaborative discussion
Version: 4.2 (Cleaned & Finalized)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- Main Attention Modules --- #

class Coord2H(nn.Module):
    """Dual-Head Coordinate Attention with selectable amplification path."""
    def __init__(self, C, groups=4, use_softmax=True, tau=1.0, kappa=1.0):
        super().__init__()
        assert C % groups == 0, f"Number of channels {C} must be divisible by groups {groups}"
        self.use_softmax = use_softmax
        self.register_buffer('tau', torch.tensor(float(tau)))
        self.register_buffer('kappa', torch.tensor(float(kappa)))

        # Frontend projection for stability
        self.proj = nn.Sequential(
            nn.Conv2d(C, C, 1, bias=False, groups=groups),
            nn.InstanceNorm2d(C, affine=True),
            nn.SiLU(inplace=True)
        )

        # H and W paths have their own dedicated convolutions (dual-head)
        self.conv_h = nn.Conv2d(C, C, 1, bias=True, groups=groups)
        self.conv_w = nn.Conv2d(C, C, 1, bias=True, groups=groups)

        # Learnable parameters for stable fusion
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta  = nn.Parameter(torch.tensor(0.5))
        self.bias  = nn.Parameter(torch.tensor(0.0))

    def _axis_softmax(self, logits, L):
        """Numerically stable softmax along a spatial axis, with mean-rescaling."""
        z = (logits / torch.clamp(self.tau, 1e-3)).to(torch.float32)
        z = z - z.amax(dim=2, keepdim=True)
        a = F.softmax(z, dim=2) * L
        return a.to(logits.dtype)

    def forward(self, x):
        B, C, H, W = x.shape

        x_h_pooled = x.mean(dim=3, keepdim=True)
        x_w_pooled = x.mean(dim=2, keepdim=True)

        x_h = self.proj(x_h_pooled)
        x_w = self.proj(x_w_pooled)

        z_h = self.conv_h(x_h)
        z_w = self.conv_w(x_w)

        if self.use_softmax:
            a_h = self._axis_softmax(z_h, H)
            a_w = self._axis_softmax(z_w.permute(0, 1, 3, 2), W).permute(0, 1, 3, 2)
        else:
            a_h = torch.sigmoid(z_h)
            a_w = torch.sigmoid(z_w)

        a_h_b = a_h.expand(-1, -1, -1, W)
        a_w_b = a_w.expand(-1, -1, H, -1)

        if self.use_softmax:
            w_lin = self.alpha * a_h_b + self.beta * a_w_b
            w = 1.0 + self.kappa * (w_lin - 1.0)
        else:
            mu_h, mu_w = 0.5, 0.5
            w_logit = self.alpha * (a_h_b - mu_h) + self.beta * (a_w_b - mu_w) + self.bias
            w = torch.sigmoid(w_logit)

        return x * w

class ChannelSpatialContextAttention(nn.Module):
    def __init__(self, dim, reduction=16, cot_k=3, cot_heads=4, cot_temperature=0.8, cot_lambda=0.7,
                 coord_heads=4, coord_use_softmax=True, coord_tau=1.0, coord_kappa=1.0):
        super(ChannelSpatialContextAttention, self).__init__()
        m = max(1, dim // reduction)
        assert m % cot_heads == 0, f"Compressed channel dim {m} must be divisible by CoT heads {cot_heads}"
        assert m % coord_heads == 0, f"Compressed channel dim {m} must be divisible by Coord heads {coord_heads}"

        self.register_buffer('cot_temperature', torch.tensor(float(cot_temperature)))
        self.register_buffer('cot_lambda', torch.tensor(float(cot_lambda)))
        self.register_buffer('gate_floor', torch.tensor(0.05)) # Bugfix: Define gate_floor

        self.compress = nn.Conv2d(dim, m, 1, bias=False)
        self.in_compress = nn.InstanceNorm2d(m, affine=True)
        
        self.coord_attention = Coord2H(m, groups=coord_heads, use_softmax=coord_use_softmax, tau=coord_tau, kappa=coord_kappa)
        self.cot_branch = self._build_cot_branch(m, cot_k, cot_heads)
        
        self.fusion = nn.Sequential(nn.Conv2d(m * 2, m, 1, bias=False), nn.InstanceNorm2d(m, affine=True), nn.ReLU(inplace=True))
        self.fusion_gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(m, 1, 1, bias=False), nn.Sigmoid())
        self.expand = nn.Conv2d(m, dim, 1, bias=False)
        
    def _build_cot_branch(self, dim, k, h):
        return nn.ModuleDict({
            'key_embed': nn.Conv2d(dim, dim, 1, bias=False, groups=h),
            'value_embed': nn.Conv2d(dim, dim, 1, bias=False, groups=h),
            'attention_embed': nn.Sequential(nn.Conv2d(dim * 2, dim, 1, bias=False, groups=h), nn.InstanceNorm2d(dim, affine=True), nn.ReLU(inplace=True), nn.Conv2d(dim, dim, 1, bias=False, groups=h)),
            'fusion_conv': nn.Conv2d(dim, dim, 1, bias=False)
        })
    
    def forward(self, x):
        x_comp = F.relu(self.in_compress(self.compress(x)), inplace=True)
        coord_out = self.coord_attention(x_comp)
        cot_out = self._cot_attention(x_comp)
        fused = self.fusion(torch.cat([coord_out, cot_out], dim=1))
        gate = self.fusion_gate(fused)
        # Apply soft floor to the gate
        output = self.expand(fused * (gate * (1 - self.gate_floor) + self.gate_floor))
        return output
    
    def _cot_attention(self, x):
        B, C, H, W = x.size()
        tau = torch.clamp(self.cot_temperature, min=1e-3)
        lam = torch.clamp(self.cot_lambda, 0.0, 1.0)

        k1 = self.cot_branch['key_embed'](x)
        v = self.cot_branch['value_embed'](x).view(B, C, -1)
        y = torch.cat([k1, x], dim=1)
        att_logits = self.cot_branch['attention_embed'](y)
        sig_att = torch.sigmoid(att_logits)
        
        z = (att_logits / tau).to(torch.float32)
        z = z.view(B, C, -1)
        z = z - z.amax(dim=-1, keepdim=True)
        sm_att = F.softmax(z, dim=-1).view(B, C, H, W)
        sm_att = sm_att.to(att_logits.dtype)
        sm_att = sm_att * (H * W)

        att = (1 - lam) * sig_att + lam * sm_att
        att_flat = att.view(B, C, -1)
        k2 = (att_flat * v).view(B, C, H, W)
        
        output = self.cot_branch['fusion_conv'](k1 + k2)
        return output

# --- Supporting Helper Modules --- #

class AttentionModuleBase(nn.Module):
    def forward(self, x):
        raise NotImplementedError

class StableGating(nn.Module):
    def __init__(self, init_value=0.1, l1_weight=1e-4):
        super(StableGating, self).__init__()
        init_logit = math.log(init_value / (1 - init_value))
        self.gate_logit = nn.Parameter(torch.ones(1) * init_logit)
        self.l1_weight = l1_weight
    def forward(self):
        gate = torch.sigmoid(self.gate_logit)
        if self.training and self.l1_weight > 0:
            if not hasattr(self, '_l1_loss'): self._l1_loss = []
            self._l1_loss.append(self.l1_weight * torch.abs(gate).mean())
        return gate
    def get_l1_loss(self):
        if hasattr(self, '_l1_loss') and self._l1_loss:
            loss = sum(self._l1_loss); self._l1_loss.clear()
            return loss
        return torch.zeros((), device=self.gate_logit.device, dtype=self.gate_logit.dtype)

class EcaChannelAttention(nn.Module):
    def __init__(self, kernel_size=5):
        super(EcaChannelAttention, self).__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y = F.adaptive_avg_pool2d(x, 1)
        y = y.squeeze(-1).squeeze(-1).unsqueeze(1)
        y = self.conv(y); y = self.sigmoid(y)
        return y.squeeze(1).unsqueeze(-1).unsqueeze(-1)

class LightweightAttention(nn.Module):
    def __init__(self, dim, eca_kernel_size=5):
        super(LightweightAttention, self).__init__()
        self.channel_attn = EcaChannelAttention(kernel_size=eca_kernel_size)
        self.spatial_attn = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.Sigmoid())
    def forward(self, x):
        ca = self.channel_attn(x); x_ca = x * ca
        avg_out = torch.mean(x_ca, dim=1, keepdim=True)
        max_out, _ = torch.max(x_ca, dim=1, keepdim=True)
        sa = self.spatial_attn(torch.cat([avg_out, max_out], dim=1))
        return x_ca * sa

class MidAttentionModule(AttentionModuleBase):
    def __init__(self, dim, init_gate=0.05, l1_weight=1e-4):
        super(MidAttentionModule, self).__init__()
        self.mid_attn = LightweightAttention(dim)
        self.gate = StableGating(init_value=init_gate, l1_weight=l1_weight)
    def forward(self, x):
        attn_out = self.mid_attn(x)
        gate_w = self.gate()
        return attn_out * gate_w

class FinalAttentionModule(AttentionModuleBase):
    def __init__(self, dim, init_gate=0.1, reduction=4, l1_weight=1e-4, config=None):
        super(FinalAttentionModule, self).__init__()
        if config is None: config = CSCAConfig()
        self.final_attn = ChannelSpatialContextAttention(dim, reduction, config.cot_k, config.cot_heads, 
                                                     config.cot_temperature, config.cot_lambda, 
                                                     config.coord_heads, config.coord_use_softmax, 
                                                     config.coord_tau, config.coord_kappa)
        self.gate = StableGating(init_value=init_gate, l1_weight=l1_weight)
    def forward(self, x):
        attn_out = self.final_attn(x)
        gate_w = self.gate()
        return attn_out * gate_w

class CSCAConfig:
    def __init__(self):
        self.use_mid_attn = True
        self.use_final_attn = True
        self.mid_init_gate = 0.05
        self.final_init_gate = 0.1
        self.reduction = 4
        self.l1_weight = 1e-4
        # CoT Branch Config
        self.cot_k = 3
        self.cot_heads = 4
        self.cot_temperature = 0.8
        self.cot_lambda = 0.7
        # Coord2H Branch Config
        self.coord_heads = 4
        self.coord_use_softmax = True
        self.coord_tau = 1.0
        self.coord_kappa = 1.0
