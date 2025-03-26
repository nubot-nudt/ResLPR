import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers

from einops import rearrange
from .wave import WaveDown
from .wave import WaveUp


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

##########################################################################
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class resblock(nn.Module):
    def __init__(self, dim):

        super(resblock, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PReLU(),
                                  nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x):
        res = self.body((x))
        res += x
        return res

##########################################################################
## Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

##########################################################################
##---------- Context Guide Module -----------------------
class ContextGuideBlock(nn.Module):
    def __init__(self, ctg_dim=64, ctg_len=5, ctg_size=64, lin_dim=96):
        super(ContextGuideBlock, self).__init__()
        self.ctg_param = nn.Parameter(torch.rand(1, ctg_len, ctg_dim, ctg_size, ctg_size))
        self.linear_layer = nn.Linear(lin_dim, ctg_len)
        self.conv3x3 = nn.Conv2d(ctg_dim, ctg_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self,x):
        B,C,H,W = x.shape
        emb = x.mean(dim=(-2,-1))
        ctg_weights = F.softmax(self.linear_layer(emb),dim=1)
        ctg = ctg_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.ctg_param.unsqueeze(0).repeat(B,1,1,1,1,1).squeeze(1)
        ctg = torch.sum(ctg, dim=1)
        ctg = F.interpolate(ctg,(H,W),mode="bilinear")
        ctg = self.conv3x3(ctg)
        return ctg

class ResLPR(nn.Module):
    def __init__(self, 
        inp_channels=2,
        out_channels=2,
        dim=48,
        num_blocks=[4,6,6,8],
        num_refinement_blocks=4,
        heads=[1,2,4,8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias',
        decoder=False,
    ):
        super(ResLPR, self).__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.decoder = decoder

        if self.decoder:
            self.ctg1 = ContextGuideBlock(ctg_dim=64, ctg_len=5, ctg_size=64, lin_dim=96)
            self.ctg2 = ContextGuideBlock(ctg_dim=128, ctg_len=5, ctg_size=32, lin_dim=192)
            self.ctg3 = ContextGuideBlock(ctg_dim=320, ctg_len=5, ctg_size=16, lin_dim=384)

        self.chnl_reduce1 = nn.Conv2d(64,64,kernel_size=1,bias=bias)
        self.chnl_reduce2 = nn.Conv2d(128,128,kernel_size=1,bias=bias)
        self.chnl_reduce3 = nn.Conv2d(320,256,kernel_size=1,bias=bias)

        self.reduce_noise_channel_1 = nn.Conv2d(dim + 64,dim,kernel_size=1,bias=bias)
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.down1_2 = WaveDown(dim)

        self.reduce_noise_channel_2 = nn.Conv2d(int(dim*2**1) + 128,int(dim*2**1),kernel_size=1,bias=bias)
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.down2_3 = WaveDown(int(dim*2**1))

        self.reduce_noise_channel_3 = nn.Conv2d(int(dim*2**2) + 256,int(dim*2**2),kernel_size=1,bias=bias)
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.down3_4 = WaveDown(int(dim*2**2)) 

        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        self.up4_3 = WaveUp(int(dim*2**2))
        
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**1)+192, int(dim*2**2), kernel_size=1, bias=bias)
        self.noise_level3 = TransformerBlock(dim=int(dim*2**2) + 512, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level3 = nn.Conv2d(int(dim*2**2)+512,int(dim*2**2),kernel_size=1,bias=bias)

        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.up3_2 = WaveUp(int(dim*2**2))
        
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.noise_level2 = TransformerBlock(dim=int(dim*2**1) + 224, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level2 = nn.Conv2d(int(dim*2**1)+224,int(dim*2**2),kernel_size=1,bias=bias)

        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.up2_1 = WaveUp(int(dim*2**1))

        self.noise_level1 = TransformerBlock(dim=int(dim*2**1)+64, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level1 = nn.Conv2d(int(dim*2**1)+64,int(dim*2**1),kernel_size=1,bias=bias)
        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
                    
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):

        inp_enc_level1 = self.patch_embed(inp_img)

        out_enc_level1 = self.encoder_level1(inp_enc_level1)  # Level 1 encoder
        
        inp_enc_level2 = self.down1_2(out_enc_level1)

        out_enc_level2 = self.encoder_level2(inp_enc_level2)  # Level 2 encoder

        inp_enc_level3 = self.down2_3(out_enc_level2)

        out_enc_level3 = self.encoder_level3(inp_enc_level3)  # Level 3 encoder

        inp_enc_level4 = self.down3_4(out_enc_level3)

        latent = self.latent(inp_enc_level4)

        if self.decoder:
            dec3_param = self.ctg3(latent)

            latent = torch.cat([latent, dec3_param], 1)
            latent = self.noise_level3(latent)
            latent = self.reduce_noise_level3(latent)
                        
        inp_dec_level3 = self.up4_3(latent)

        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)

        out_dec_level3 = self.decoder_level3(inp_dec_level3)  # Level 3 decoder
        if self.decoder:
            dec2_param = self.ctg2(out_dec_level3)
            out_dec_level3 = torch.cat([out_dec_level3, dec2_param], 1)
            out_dec_level3 = self.noise_level2(out_dec_level3)
            out_dec_level3 = self.reduce_noise_level2(out_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)

        out_dec_level2 = self.decoder_level2(inp_dec_level2)  # Level 2 decoder
        if self.decoder:
           
            dec1_param = self.ctg1(out_dec_level2)
            out_dec_level2 = torch.cat([out_dec_level2, dec1_param], 1)
            out_dec_level2 = self.noise_level1(out_dec_level2)
            out_dec_level2 = self.reduce_noise_level1(out_dec_level2)
        
        inp_dec_level1 = self.up2_1(out_dec_level2)
        
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        
        out_dec_level1 = self.decoder_level1(inp_dec_level1)  # Level 1 decoder

        out_dec_level1 = self.refinement(out_dec_level1)

        out_dec_level1 = self.output(out_dec_level1) + inp_img
        return out_dec_level1