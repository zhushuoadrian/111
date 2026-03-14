# -----------------------------------------------------------------------------------
# 🚀 融合版：Original Backbone + WaveMamba (DWT 分频) + CNN Fourier
# -----------------------------------------------------------------------------------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange

# 🌟 动态导入 Mamba 模块
try:
    from .vmamba import SS2D  
except ImportError:
    from vmamba import SS2D   

# ===================================================================================
# 1. 核心组件：WaveMamba 频域分离 (DWT / IWT)
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
    def forward(self, x):
        return dwt_init(x)

class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False
    def forward(self, x):
        return iwt_init(x)

# ===================================================================================
# 2. Mamba 处理块：专攻低频 (LFSSBlock)
# ===================================================================================
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
        x = F.gelu(x1) * x2
        x = self.conv3(x)
        return x

class LFSSBlock(nn.Module):
    def __init__(self, hidden_dim, drop_path=0.):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, d_state=16)
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
# 3. 你的原版基础组件 (GatedRefine, SKFF, 下/上采样)
# ===================================================================================
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class GatedRefine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.project_in = nn.Conv2d(dim, dim * 2, 1)
        self.sg = SimpleGate()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dim, dim, 1))
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

class SKFF(nn.Module):
    def __init__(self, in_channels, height=2, reduction=2, bias=False): 
        super(SKFF, self).__init__()
        self.height = height
        d = max(int(in_channels / reduction), 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, bias=bias), nn.ReLU())
        self.fcs = nn.ModuleList([nn.Conv2d(d, in_channels, 1, bias=bias) for _ in range(self.height)])
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats = inp_feats[0].shape[1]
        inp_feats = torch.cat(inp_feats, dim=1).view(batch_size, self.height, n_feats, inp_feats[0].shape[2], inp_feats[0].shape[3])
        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)
        attention_vectors = torch.cat([fc(feats_Z) for fc in self.fcs], dim=1).view(batch_size, self.height, n_feats, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        return torch.sum(inp_feats * attention_vectors, dim=1)

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

# ===================================================================================
# 4. 你的原版核心 CNN 模块 (Local + Global Fourier)
# ===================================================================================
class OurTokenMixer_For_Local(nn.Module):
    def __init__(self, dim, kernel_size=[1,3,5,7], se_ratio=4, local_size=8, scale_ratio=2, spilt_num=4):
        super(OurTokenMixer_For_Local, self).__init__()
        self.dim = dim
        self.dim_sp = dim*scale_ratio//spilt_num
        self.conv_init = nn.Sequential(nn.Conv2d(dim, dim*scale_ratio, 1), nn.GELU())
        self.conv_fina = nn.Sequential(nn.Conv2d(dim*scale_ratio, dim, 1), nn.GELU())
        self.conv1_1 = nn.Sequential(nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=kernel_size[1], padding=kernel_size[1] // 2, groups=self.dim_sp, padding_mode='reflect'), nn.GELU())
        self.conv1_2 = nn.Sequential(nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=kernel_size[2], padding=kernel_size[2] // 2, groups=self.dim_sp, padding_mode='reflect'), nn.GELU())
        self.conv1_3 = nn.Sequential(nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=kernel_size[3], padding=kernel_size[3] // 2, groups=self.dim_sp, padding_mode='reflect'), nn.GELU())

    def forward(self, x):
        x = self.conv_init(x)
        x = list(torch.split(x, self.dim_sp, dim=1))
        x[1] = self.conv1_1(x[1])
        x[2] = self.conv1_2(x[2])
        x[3] = self.conv1_3(x[3])
        x = torch.cat(x, dim=1)
        return self.conv_fina(x)

class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.conv_layer1 = nn.Sequential(torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2, kernel_size=1, stride=1, padding=0, groups=self.groups,bias=True), nn.GELU())
        self.bn1 = torch.nn.BatchNorm2d(out_channels * 2)
    def forward(self, x):
        batch, c, h, w = x.size()
        ffted = torch.fft.rfft2(x, norm='ortho')
        x_fft_real = torch.unsqueeze(torch.real(ffted), dim=-1)
        x_fft_imag = torch.unsqueeze(torch.imag(ffted), dim=-1)
        ffted = torch.cat((x_fft_real, x_fft_imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        ffted = self.conv_layer1(self.bn1(ffted))
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()
        ffted = torch.view_as_complex(ffted)
        output = torch.fft.irfft2(ffted, s=(h, w), norm='ortho')
        return output

class OurTokenMixer_For_Gloal(nn.Module):
    def __init__(self, dim, kernel_size=[1,3,5,7], se_ratio=4, local_size=8, scale_ratio=2, spilt_num=4):
        super(OurTokenMixer_For_Gloal, self).__init__()
        self.dim = dim
        self.conv_init = nn.Sequential(nn.Conv2d(dim, dim*2, 1), nn.GELU())
        self.conv_fina = nn.Sequential(nn.Conv2d(dim*2, dim, 1), nn.GELU())
        self.FFC = FourierUnit(self.dim*2, self.dim*2)
    def forward(self, x):
        x = self.conv_init(x)
        x0 = x
        x = self.FFC(x)
        return self.conv_fina(x+x0)

class OurMixer(nn.Module):
    def __init__(self, dim, token_mixer_for_local=OurTokenMixer_For_Local, token_mixer_for_gloal=OurTokenMixer_For_Gloal, mixer_kernel_size=[1,3,5,7], local_size=8):
        super(OurMixer, self).__init__()
        self.dim = dim
        self.mixer_local = token_mixer_for_local(dim=self.dim, kernel_size=mixer_kernel_size, se_ratio=8, local_size=local_size)
        self.mixer_gloal = token_mixer_for_gloal(dim=self.dim, kernel_size=mixer_kernel_size, se_ratio=8, local_size=local_size)
        self.ca_conv = nn.Sequential(nn.Conv2d(2*dim, dim, 1), nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, padding_mode='reflect'), nn.GELU())
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dim, dim // 4, kernel_size=1), nn.GELU(), nn.Conv2d(dim // 4, dim, kernel_size=1), nn.Sigmoid())
        self.conv_init = nn.Sequential(nn.Conv2d(dim, dim * 2, 1), nn.GELU())
    def forward(self, x):
        x = self.conv_init(x)
        x = list(torch.split(x, self.dim, dim=1))
        x_local = self.mixer_local(x[0])
        x_gloal = self.mixer_gloal(x[1])
        x = torch.cat([x_local, x_gloal], dim=1)
        x = self.ca_conv(x)
        return self.ca(x) * x

class OurBlock(nn.Module):
    def __init__(self, dim, norm_layer=nn.BatchNorm2d, token_mixer=OurMixer, kernel_size=[1,3,5,7], local_size=8):
        super(OurBlock, self).__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.mixer = token_mixer(dim=dim, mixer_kernel_size=kernel_size, local_size=local_size)
        self.ffn = OurTokenMixer_For_Local(dim=dim, kernel_size=kernel_size)
    def forward(self, x):
        x = self.mixer(self.norm1(x)) + x
        x = self.ffn(self.norm2(x)) + x
        return x

# ===================================================================================
# ⭐ 5. 核心重构：WaveMamba 并行模块 (完美替代原 ParallelStage) ⭐
# ===================================================================================
class WaveParallelStage(nn.Module):
    """
    革命性地将原始笨重的 Swin + CNN 改为了 小波分离 (DWT) 架构：
    1. 极低显存：DWT 让内部特征图变为 H/2, W/2。
    2. Mamba 处理低频（全局光照）。
    3. 你原来的 CNN + Fourier 处理高频（局部纹理）。
    """
    def __init__(self, depth, in_channels, mixer_kernel_size=[1,3,5,7], local_size=8):
        super(WaveParallelStage, self).__init__()
        self.dwt = DWT()
        self.iwt = IWT()
        
        # 1. 低频分支：纯正 Mamba 模块 (吃 LL 波段)
        self.mamba_branch = nn.Sequential(*[
            LFSSBlock(hidden_dim=in_channels) for _ in range(depth)
        ])
        
        # 2. 高频分支：你原版的 CNN+Fourier (吃 HL, LH, HH 波段)
        self.h_fusion = SKFF(in_channels, height=3, reduction=2) 
        self.cnn_branch = nn.Sequential(*[
            OurBlock(dim=in_channels, norm_layer=nn.BatchNorm2d, token_mixer=OurMixer,
                     kernel_size=mixer_kernel_size, local_size=local_size)
            for _ in range(depth)
        ])
        
        # 将融合后的单通道高频扩展为 3 个通道，配合低频做逆小波变换(IWT)
        self.h_out_conv = nn.Conv2d(in_channels, in_channels * 3, 3, 1, 1)
        
        # 特征平滑融合
        self.refine = GatedRefine(in_channels)

    def forward(self, x):
        input_tensor = x
        
        # 1. 小波分解 (得到一半尺寸的图)
        x_LL, x_HL, x_LH, x_HH = self.dwt(x)
        
        # 2. 低频进 Mamba
        B, C, H2, W2 = x_LL.shape
        x_l = rearrange(x_LL, "b c h w -> b (h w) c").contiguous()
        for m_blk in self.mamba_branch:
            x_l = m_blk(x_l, [H2, W2])
        x_l = rearrange(x_l, "b (h w) c -> b c h w", h=H2, w=W2).contiguous()
        
        # 3. 高频进 CNN
        x_h = self.h_fusion([x_HL, x_LH, x_HH])
        for c_blk in self.cnn_branch:
            x_h = c_blk(x_h)
            
        # 4. 重构：恢复原本的分辨率 (H, W)
        x_h_expand = self.h_out_conv(x_h)
        x_fused_dwt = self.iwt(torch.cat([x_l, x_h_expand], dim=1))
        
        # 5. 收尾提纯
        out = self.refine(x_fused_dwt)
        
        return out + input_tensor

# ===================================================================================
# 6. 主干网络 (彻底摒弃 RSTB，改为小波 Mamba 驱动)
# ===================================================================================
class Backbone_new(nn.Module):
    def __init__(self, in_chans=3, out_chans=3, patch_size=1,
                 embed_dim=[48, 96, 192, 96, 48], depth=[2, 2, 2, 2, 2],
                 local_size=[4, 4, 4, 4 ,4],
                 act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
                 norm_layer_transformer=nn.LayerNorm, embed_kernel_size=3,
                 downsample_kernel_size=None, upsample_kernel_size=None, **kwargs):
        super(Backbone_new, self).__init__()

        if downsample_kernel_size is None: downsample_kernel_size = 4
        if upsample_kernel_size is None: upsample_kernel_size = 4

        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim[0], kernel_size=embed_kernel_size)
        
        # ⭐ 核心替换：全部换成了 WaveParallelStage
        self.layer1 = WaveParallelStage(depth=depth[0], in_channels=embed_dim[0], mixer_kernel_size=[1, 3, 5, 7], local_size=local_size[0])
        self.skip1 = SKFF(in_channels=embed_dim[0], reduction=2) 
        self.refine1 = GatedRefine(embed_dim[0])
        self.downsample1 = DownSample(input_dim=embed_dim[0], output_dim=embed_dim[1], kernel_size=downsample_kernel_size, stride=2)
        
        self.layer2 = WaveParallelStage(depth=depth[1], in_channels=embed_dim[1], mixer_kernel_size=[1, 3, 5, 7], local_size=local_size[1])
        self.skip2 = SKFF(in_channels=embed_dim[1], reduction=2)
        self.refine2 = GatedRefine(embed_dim[1])
        self.downsample2 = DownSample(input_dim=embed_dim[1], output_dim=embed_dim[2], kernel_size=downsample_kernel_size, stride=2)
        
        self.layer3 = WaveParallelStage(depth=depth[2], in_channels=embed_dim[2], mixer_kernel_size=[1, 3, 5, 7], local_size=local_size[2])
        self.upsample1 = PatchUnEmbed_for_upsample(patch_size=2, embed_dim=embed_dim[2], out_dim=embed_dim[3])
        
        self.layer4 = WaveParallelStage(depth=depth[3], in_channels=embed_dim[3], mixer_kernel_size=[1, 3, 5, 7], local_size=local_size[3])
        self.upsample2 = PatchUnEmbed_for_upsample(patch_size=2, embed_dim=embed_dim[3], out_dim=embed_dim[4])
        
        self.layer5 = WaveParallelStage(depth=depth[4], in_channels=embed_dim[4], mixer_kernel_size=[1, 3, 5, 7], local_size=local_size[4])
        self.patch_unembed = PatchUnEmbed(patch_size=patch_size, out_chans=out_chans, embed_dim=embed_dim[4], kernel_size=3)

    def forward(self, x):
        copy0 = x
        x = self.patch_embed(x)
        x = self.layer1(x)
        copy1 = x

        x = self.downsample1(x)
        x = self.layer2(x)
        copy2 = x

        x = self.downsample2(x)
        x = self.layer3(x)
        
        x = self.upsample1(x)
        x = self.skip2([x, copy2])
        x = self.refine2(x)
        x = self.layer4(x)
        
        x = self.upsample2(x)
        x = self.skip1([x, copy1])
        x = self.refine1(x)
        x = self.layer5(x)
        
        x = self.patch_unembed(x)
        return copy0 + x

# -----------------------------------------------------------------------------------
# 7. 模型定义
# -----------------------------------------------------------------------------------
def sfhformer_t(**kwargs): return Backbone_new(embed_dim=[24, 48, 96, 48, 24], depth=[1, 1, 2, 1, 1], **kwargs)
def sfhformer_lol_s(**kwargs): return Backbone_new(embed_dim=[24, 48, 96, 48, 24], depth=[1, 1, 2, 1, 1], **kwargs)
def sfhformer_lol_m(**kwargs): return Backbone_new(embed_dim=[24, 48, 96, 48, 24], depth=[2, 2, 4, 2, 2], **kwargs)
def sfhformer_lol_l(**kwargs): return Backbone_new(embed_dim=[24, 48, 96, 48, 24], depth=[4, 4, 8, 4, 4], **kwargs)

if __name__ == "__main__":
    print("🔍 Testing Model: WaveMamba Parallel (DWT + Mamba Global + CNN Fourier Local)")
    model = sfhformer_lol_s()
    x = torch.randn(2, 3, 64, 64)
    if torch.cuda.is_available():
        model = model.cuda()
        x = x.cuda()
    y = model(x)
    print(f"✅ Forward Pass: {x.shape} -> {y.shape}")