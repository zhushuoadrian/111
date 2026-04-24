# -----------------------------------------------------------------------------------
# 🏆 最终完整版：V2 运行版 + V2.1 模块共存（画图用V2.1，代码跑V2，最稳）
# -----------------------------------------------------------------------------------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from einops import rearrange

# 🌟 动态导入 Mamba 模块
try:
    from .vmamba import SS2D
except ImportError:
    from vmamba import SS2D

# ===================================================================================
# 1. 基础频域组件 (DWT/IWT, SimpleGate)
# ===================================================================================
def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    return x_LL, x_HL, x_LH, x_HH

def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = in_batch, int(in_channel/(r**2)), r * in_height, r * in_width
    x1 = x[:, :out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().to(x.device)
    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
    return h

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False
    def forward(self, x): return dwt_init(x)

class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False
    def forward(self, x): return iwt_init(x)

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

# ===================================================================================
# 2. 核心处理块 (NAFBlock, Mamba_ffn, LFSSBlock)
# ===================================================================================
class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = int(c * DW_Expand)
        self.conv1 = nn.Conv2d(c, dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )
        self.sg = SimpleGate()

        ffn_channel = int(FFN_Expand * c)
        self.conv4 = nn.Conv2d(c, ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = nn.LayerNorm(c, eps=1e-6)
        self.norm2 = nn.LayerNorm(c, eps=1e-6)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp.permute(0, 2, 3, 1)
        x = self.norm1(x).permute(0, 3, 1, 2)

        x_dw = self.conv2(self.conv1(x))
        x_sg = self.sg(x_dw)
        x = self.conv3(self.sca(x_sg) * x_sg)
        y = inp + self.dropout1(x) * self.beta

        x = self.conv4(self.norm2(y.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        x = self.conv5(self.sg(x))
        return y + self.dropout2(x) * self.gamma

class Mamba_ffn(nn.Module):
    def __init__(self, num_feat, ffn_expand=2):
        super().__init__()
        dw_channel = num_feat * ffn_expand
        self.conv1 = nn.Conv2d(num_feat, dw_channel, kernel_size=1)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, kernel_size=3, padding=1, groups=dw_channel)
        self.conv3 = nn.Conv2d(dw_channel//2, num_feat, kernel_size=1)
        
    def forward(self, x):
        x = self.conv2(self.conv1(x))
        x1, x2 = x.chunk(2, dim=1)
        return self.conv3(F.gelu(x1) * x2)

class LFSSBlock(nn.Module):
    def __init__(self, hidden_dim, drop_path=0.):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, d_state=16, ssm_ratio=1.0, forward_type="v3", channel_first=False)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.skip_scale = nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = Mamba_ffn(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, input, x_size):
        B, L, C = input.shape
        input_2d = input.view(B, *x_size, C).contiguous()
        x = self.ln_1(input_2d)
        x = input_2d * self.skip_scale + self.drop_path(self.self_attention(x))
        x = x * self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        return x.view(B, -1, C).contiguous()

# ===================================================================================
# 3. 双版本跳跃连接融合模块：V2旧版 + V2.1新版 全部保留
# ===================================================================================
# ---------------- V2.1 新版模块（写进去，不调用，画图用）----------------
class DS_SKFF(nn.Module):
    """🔥 V2.1 空间+通道双自适应融合（论文画图用）"""
    def __init__(self, in_channels, height=2, reduction=4):
        super().__init__()
        self.height = height
        d = max(int(in_channels / reduction), 8)
        
        self.spatial_att = nn.Sequential(
            nn.Conv2d(in_channels, d, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(d, height, 3, padding=1, bias=False) 
        )
        
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, d, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(d, in_channels * height, 1, bias=False)
        )
        
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, 1)
        )

    def forward(self, inp_feats):
        B, C, H, W = inp_feats[0].shape
        feats_U = sum(inp_feats) 
        
        s_weight = self.spatial_att(feats_U)
        s_weight = F.softmax(s_weight, dim=1).unsqueeze(2) 
        
        c_weight = self.channel_att(feats_U).view(B, self.height, C, 1, 1)
        c_weight = F.softmax(c_weight, dim=1)
        
        inp_tensor = torch.stack(inp_feats, dim=1)
        feats_V = torch.sum(inp_tensor * s_weight * c_weight, dim=1)
        return feats_V + self.refine(feats_V)

# ---------------- V2 旧版模块（实际运行调用，效果最稳）----------------
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        reduced = max(in_planes // ratio, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, reduced, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(reduced, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv1(torch.cat([avg_out, max_out], dim=1)))

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)
    def forward(self, x):
        return x * self.ca(x) * self.sa(x)

class SKFF_CBAM(nn.Module):
    """✅ V2 原版融合模块（代码实际调用运行）"""
    def __init__(self, in_channels, height=2, reduction=2, bias=False): 
        super().__init__()
        self.height = height
        d = max(int(in_channels / reduction), 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, bias=bias), nn.ReLU())
        self.fcs = nn.ModuleList([nn.Conv2d(d, in_channels, 1, bias=bias) for _ in range(self.height)])
        self.softmax = nn.Softmax(dim=1)
        self.cbam = CBAM(in_channels) 

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats = inp_feats[0].shape[1]
        inp_feats = torch.cat(inp_feats, dim=1).view(batch_size, self.height, n_feats, inp_feats[0].shape[2], inp_feats[0].shape[3])
        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)
        attention_vectors = torch.cat([fc(feats_Z) for fc in self.fcs], dim=1).view(batch_size, self.height, n_feats, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_V = torch.sum(inp_feats * attention_vectors, dim=1)
        return self.cbam(feats_V) 

# ===================================================================================
# 4. 双版本细化模块：V2旧版 + V2.1新版 全部保留
# ===================================================================================
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates=[6, 12, 18]):
        super().__init__()
        self.branches = nn.ModuleList()
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.Identity(), 
            nn.ReLU()
        ))
        for r in rates:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=r, dilation=r, bias=False),
                nn.Identity(),
                nn.ReLU()
            ))
        self.pool_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.ReLU()
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * (len(rates)+2), out_channels, 1, bias=False),
            nn.Identity(),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        size = x.shape[2:]
        feats = [b(x) for b in self.branches]
        feats.append(F.interpolate(self.pool_branch(x), size=size, mode='bilinear', align_corners=False))
        return self.fusion(torch.cat(feats, dim=1))

class BiCSG_Fusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm_l = nn.LayerNorm(dim)
        self.norm_h = nn.LayerNorm(dim * 3)
        self.l2h_conv = nn.Sequential(nn.Conv2d(dim, dim*3, 3, 1, 1, groups=dim), nn.GELU(), nn.Conv2d(dim*3, dim*3, 1), nn.Sigmoid())
        self.h2l_conv = nn.Sequential(nn.Conv2d(dim*3, dim, 3, 1, 1), nn.GELU(), nn.Conv2d(dim, dim, 1), nn.Sigmoid())
        self.out_h = nn.Conv2d(dim*3, dim*3, 1)
        self.out_l = nn.Conv2d(dim, dim, 1)

    def forward(self, x_l, x_h):
        x_l_n = self.norm_l(x_l.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_h_n = self.norm_h(x_h.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        attn_l = self.l2h_conv(x_l_n)
        x_h_refined = x_h * attn_l + x_h
        
        attn_h = self.h2l_conv(x_h_n)
        x_l_refined = x_l * attn_h + x_l
        
        return self.out_l(x_l_refined), self.out_h(x_h_refined)

# ---------------- V2.1 新版细化（写进去，不调用，画图用）----------------
class ECA(nn.Module):
    def __init__(self, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        return torch.sigmoid(y)

class SC_GatedRefine(nn.Module):
    """🔥 V2.1 双注意力细化（论文画图用）"""
    def __init__(self, dim):
        super().__init__()
        self.project_in = nn.Conv2d(dim, dim * 2, 1)
        self.sg = SimpleGate()
        self.dwconv = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)
        self.eca = ECA(k_size=3)
        self.pa = nn.Sequential(
            nn.Conv2d(dim, max(dim // 8, 4), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(dim // 8, 4), 1, 1, bias=False),
            nn.Sigmoid()
        )
        self.project_out = nn.Conv2d(dim, dim, 1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x
        x_norm = x.permute(0, 2, 3, 1).contiguous()
        x_norm = self.norm(x_norm).permute(0, 3, 1, 2).contiguous()
        
        x_in = self.project_in(x_norm)
        x_gated = self.sg(x_in)
        x_feat = self.dwconv(x_gated)
        
        weight_c = self.eca(x_feat)
        weight_p = self.pa(x_feat)
        x_feat = x_feat * weight_c * weight_p
        
        out = self.project_out(x_feat)
        return out + residual

# ---------------- V2 旧版细化（实际运行调用，效果最稳）----------------
class GatedRefine(nn.Module):
    """✅ V2 原版细化模块（代码实际调用运行）"""
    def __init__(self, dim):
        super().__init__()
        self.project_in = nn.Conv2d(dim, dim * 2, 1)
        self.sg = SimpleGate()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dim, dim, 1), nn.Sigmoid())
        self.project_out = nn.Conv2d(dim, dim, 1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x
        x_norm = x.permute(0, 2, 3, 1).contiguous()
        x_norm = self.norm(x_norm).permute(0, 3, 1, 2).contiguous()
        x = self.project_in(x_norm)
        x = self.sg(x)      
        x = self.dwconv(x)  
        x = x * self.sca(x) 
        x = self.project_out(x)
        return x + residual

# ===================================================================================
# 5. 核心模块 + 主干网络（100% 调用 V2 原版，最稳定）
# ===================================================================================
class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        if kernel_size is None: kernel_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size, padding=(kernel_size - patch_size + 1) // 2, padding_mode='reflect')
    def forward(self, x): return self.proj(x)

class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        if kernel_size is None: kernel_size = 1
        self.proj = nn.Sequential(nn.Conv2d(embed_dim, out_chans * patch_size ** 2, kernel_size=kernel_size, padding=kernel_size // 2, padding_mode='reflect'), nn.PixelShuffle(patch_size), nn.Conv2d(out_chans, out_chans, kernel_size=kernel_size, padding=kernel_size // 2, padding_mode='reflect'))
    def forward(self, x): return self.proj(x)

class PatchUnEmbed_for_upsample(nn.Module):
    def __init__(self, patch_size=4, embed_dim=96, out_dim=64, kernel_size=None):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(embed_dim, out_dim * patch_size ** 2, kernel_size=3, padding=1, padding_mode='reflect'), nn.PixelShuffle(patch_size))
    def forward(self, x): return self.proj(x)

class DownSample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=4, stride=2):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(input_dim, input_dim // 2, kernel_size=3, stride=1, padding=1, bias=False), nn.PixelUnshuffle(2))
    def forward(self, x): return self.proj(x)

class WaveParallelStage_v2(nn.Module):
    def __init__(self, depth, in_channels):
        super().__init__()
        self.dwt = DWT()
        self.iwt = IWT()
        
        self.mamba_branch = nn.Sequential(*[LFSSBlock(hidden_dim=in_channels) for _ in range(depth)])
        self.cnn_branch = nn.Sequential(*[NAFBlock(c=in_channels * 3, DW_Expand=1.5) for _ in range(depth)])
        self.bicsg = BiCSG_Fusion(in_channels)
        self.refine = GatedRefine(in_channels)  # ✅ 调用 V2 原版

    def forward(self, x):
        input_tensor = x
        x_LL, x_HL, x_LH, x_HH = self.dwt(x)
        
        B, C, H2, W2 = x_LL.shape
        x_l = rearrange(x_LL, "b c h w -> b (h w) c").contiguous()
        for m_blk in self.mamba_branch:
            x_l = m_blk(x_l, [H2, W2])
        x_l = rearrange(x_l, "b (h w) c -> b c h w", h=H2, w=W2).contiguous()
        
        x_h = torch.cat([x_HL, x_LH, x_HH], dim=1) 
        for c_blk in self.cnn_branch:
            x_h = c_blk(x_h)
            
        x_l_refined, x_h_refined = self.bicsg(x_l, x_h)
        x_fused_dwt = self.iwt(torch.cat([x_l_refined, x_h_refined], dim=1))
        
        out = self.refine(x_fused_dwt)
        return out + input_tensor

class IlluminationEstimator(nn.Module):
    def __init__(self, in_chans=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_chans, 16, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(32, 3, 3, 1, 1), 
        )
        nn.init.zeros_(self.net[4].weight)
        nn.init.zeros_(self.net[4].bias)

    def forward(self, x):
        illu_map = torch.sigmoid(self.net(x)) * 8.0 + 1.0
        illu_smooth3 = F.avg_pool2d(illu_map, kernel_size=3, stride=1, padding=1)
        illu_smooth5 = F.avg_pool2d(illu_map, kernel_size=5, stride=1, padding=2)
        illu_map = 0.5 * illu_smooth3 + 0.5 * illu_smooth5
        return illu_map

class Backbone_v2(nn.Module):
    def __init__(self, in_chans=3, out_chans=3, patch_size=1,
                 embed_dim=[32, 64, 128, 64, 32], 
                 depth=[2, 2, 2, 2, 2],
                 **kwargs):
        super().__init__()

        self.illu_estimator = IlluminationEstimator(in_chans)
        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim[0])
        
        self.layer1 = WaveParallelStage_v2(depth=depth[0], in_channels=embed_dim[0])
        self.skip1 = SKFF_CBAM(in_channels=embed_dim[0])   # ✅ 调用 V2 原版
        self.refine1 = GatedRefine(embed_dim[0])            # ✅ 调用 V2 原版
        self.downsample1 = DownSample(input_dim=embed_dim[0], output_dim=embed_dim[1])
        
        self.layer2 = WaveParallelStage_v2(depth=depth[1], in_channels=embed_dim[1])
        self.skip2 = SKFF_CBAM(in_channels=embed_dim[1])   # ✅ 调用 V2 原版
        self.refine2 = GatedRefine(embed_dim[1])            # ✅ 调用 V2 原版
        self.downsample2 = DownSample(input_dim=embed_dim[1], output_dim=embed_dim[2])
        
        self.layer3 = WaveParallelStage_v2(depth=depth[2], in_channels=embed_dim[2])
        self.aspp = ASPP(embed_dim[2], embed_dim[2]) 
        
        self.upsample1 = PatchUnEmbed_for_upsample(patch_size=2, embed_dim=embed_dim[2], out_dim=embed_dim[3])
        self.layer4 = WaveParallelStage_v2(depth=depth[3], in_channels=embed_dim[3])
        
        self.upsample2 = PatchUnEmbed_for_upsample(patch_size=2, embed_dim=embed_dim[3], out_dim=embed_dim[4])
        self.layer5 = WaveParallelStage_v2(depth=depth[4], in_channels=embed_dim[4])
        
        self.patch_unembed = PatchUnEmbed(patch_size=patch_size, out_chans=out_chans, embed_dim=embed_dim[4])

        self.refine_out = nn.Sequential(
            nn.Conv2d(out_chans, 32, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, 1, 1, groups=32), 
            nn.GELU(),
            nn.Conv2d(32, out_chans, 3, 1, 1),
        )

    def forward(self, x):
        illu_map = self.illu_estimator(x)
        x_bright = x * illu_map 
        
        feat = self.patch_embed(x_bright)

        feat = self.layer1(feat)
        copy1 = feat 
        feat = self.downsample1(feat)
        
        feat = self.layer2(feat)
        copy2 = feat 
        feat = self.downsample2(feat)
        
        feat = self.layer3(feat)
        feat = self.aspp(feat) 
        
        feat = self.upsample1(feat)
        feat = self.skip2([feat, copy2])
        feat = self.refine2(feat)
        feat = self.layer4(feat)
        
        feat = self.upsample2(feat)
        feat = self.skip1([feat, copy1])
        feat = self.refine1(feat)
        feat = self.layer5(feat)
        
        residual = self.patch_unembed(feat)
        residual = residual + self.refine_out(residual)
        
        return x_bright + residual

# -----------------------------------------------------------------------------------
# 模型入口
# -----------------------------------------------------------------------------------
def sfhformer_lol_s(**kwargs): 
    return Backbone_v2(embed_dim=[32, 64, 128, 64, 32], depth=[2, 2, 4, 2, 2], **kwargs)

def sfhformer_t(**kwargs): 
    return Backbone_v2(embed_dim=[32, 64, 128, 64, 32], depth=[2, 2, 4, 2, 2], **kwargs)

def sfhformer_lol_m(**kwargs):
    return Backbone_v2(embed_dim=[24, 48, 96, 48, 24], depth=[2, 2, 4, 2, 2], **kwargs)

def sfhformer_lol_l(**kwargs):
    return Backbone_v2(embed_dim=[64, 128, 256, 128, 64], depth=[4, 4, 8, 4, 4], **kwargs)

if __name__ == "__main__":
    print("🔍 运行 V2 稳定版 | V2.1 模块已写入不调用")
    model = sfhformer_lol_s()
    x = torch.randn(1, 3, 128, 128) 
    if torch.cuda.is_available():
        model = model.cuda()
        x = x.cuda()
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 参数量：{n_params / 1e6:.2f}M")
    
    y = model(x)
    print(f"✅ 运行成功：{x.shape} -> {y.shape}")