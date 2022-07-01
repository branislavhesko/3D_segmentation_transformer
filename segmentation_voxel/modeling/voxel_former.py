from math import sqrt

import einops
import torch
import torch.nn as nn



class VoxelEmbedding(nn.Module):
    
    def __init__(self, in_features, embed_dim, stride) -> None:
        super().__init__()
        self.stride = stride
        self.linear = nn.Linear(in_features * (self.stride ** 3), out_features=embed_dim)
    
    def forward(self, volume):
        feats = einops.rearrange(volume, "b c (i x) (j y) (k z) -> b (c i j k) (x y z)", 
                         i=self.stride, j=self.stride, k=self.stride)
        feats = einops.rearrange(feats, "b c e-> b e c")
        return self.linear(feats)
    

class DeconvLayer(nn.Sequential):
    
    def __init__(self, in_features, out_features, kernel_size, stride=2, use_conv=True) -> None:
        super().__init__()
        self.layers = [nn.ConvTranspose3d(in_features, out_features, kernel_size, stride)]
        if use_conv:
            self.layers.append(ConvBatchNormRelu(in_features, in_features, kernel_size=3, stride=1, padding=1))
        super().__init__(*self.layers)
    
class ConvBatchNormRelu(nn.Sequential):
    def __init__(
            self, 
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            activation=nn.ReLU,
            norm=nn.BatchNorm3d
        ):
        self.layers = [nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding), ]
        if norm is not None:
            self.layers += [norm(out_channels), ]
        if activation is not None:
            self.layers += [activation(), ]
        super().__init__(*self.layers)


class MLP(nn.Sequential):
    def __init__(self, embed_size, expansion=4, dropout_layer=nn.Dropout, dropout_probability=0.1):
        super().__init__(
            *[
                nn.Linear(embed_size, expansion * embed_size),
                dropout_layer(p=dropout_probability),
                nn.GELU(),
                nn.Linear(expansion * embed_size, embed_size),
                dropout_layer(p=dropout_probability)
            ]
        )
        

class ResidualAdd(nn.Module):
    
    def __init__(self, blocks):
        super().__init__()
        self.blocks = blocks
        
    def forward(self, x):
        return x + self.blocks(x)
    
    
class PreNorm(nn.Module):
    
    def __init__(self, block, dim) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.block = block
        
    def forward(self, x):
        return self.norm(self.block(x))
        
    
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, embed_size, num_heads, dropout_probability=0.1):
        super().__init__()
        self.projection = nn.Linear(embed_size, embed_size * 3)
        self.final_projection = nn.Linear(embed_size, embed_size)
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.dropout = nn.Dropout(p=dropout_probability)

    def forward(self, x):
        assert len(x.shape) == 3
        qkv = self.projection(x)
        
        queries, keys, values = qkv.chunk(3, dim=2)
        keys = einops.rearrange(keys, "b n (h e) -> b n h e", h=self.num_heads)
        queries = einops.rearrange(queries, "b n (h e) -> b n h e", h=self.num_heads)
        values = einops.rearrange(values, "b n (h e) -> b n h e", h=self.num_heads)
        energy_term = torch.einsum("bqhe, bkhe -> bqhk", queries, keys)
        divider = sqrt(self.embed_size)
        mh_out = torch.softmax(energy_term, -1)
        out = torch.einsum('bihv, bvhd -> bihd ', mh_out / divider, values)
        out = einops.rearrange(out, "b n h e -> b n (h e)")
        return self.dropout(self.final_projection(out))
    

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_size, num_heads, dropout_probability=0.1):
        super().__init__()
        self.attention = ResidualAdd(PreNorm(
            MultiHeadAttention(embed_size, num_heads, dropout_probability), dim=embed_size))
        self.feed_forward = ResidualAdd(PreNorm(MLP(embed_size), dim=embed_size))

        
    def forward(self, x):
        out = self.attention(x)
        out = self.feed_forward(out)
        return out + x


class SegmentationTransformer3D(nn.Module):
    
    def __init__(self) -> None:
        pass
    
    def forward(self, x):
        pass
