import pytest
from segmentation_voxel.modeling.voxel_former import (
    ConvBatchNormRelu, PreNorm, MLP, MultiHeadAttention, ResidualAdd, VoxelEmbedding)
import torch


class TestModel:
    
    @pytest.mark.parametrize("inputs, embed_dim", [
        (torch.rand(2, 3, 16, 16, 16), 256),
        (torch.rand(2, 4, 32, 32, 32), 384),
        (torch.rand(2, 1, 128, 128, 128), 384),
        (torch.rand(2, 3, 64, 64, 64), 512)
    ])
    def test_voxel_embedding(self, inputs, embed_dim):
        stride = 8
        voxel_embedding = VoxelEmbedding(inputs.shape[1], embed_dim, stride=stride)
        model_out = voxel_embedding(inputs)
        assert model_out.shape[2] == embed_dim
        assert len(model_out.shape) == 3
        assert model_out.shape[1] == inputs.shape[2] * inputs.shape[3] * inputs.shape[4] // (stride ** 3)

    @pytest.mark.parametrize("inputs, in_channels, out_channels", [
        (torch.rand(2, 3, 16, 16, 16), 3, 64),
        (torch.rand(2, 16, 32, 32, 32), 16, 128),
        (torch.rand(2, 32, 32, 32, 32), 32, 256),
        
    ])
    def test_conv_batch_norm_relu(self, inputs, in_channels, out_channels):
        op = ConvBatchNormRelu(in_channels, out_channels)
        out = op(inputs)
        assert out.shape == (inputs.shape[0], out_channels, inputs.shape[2], inputs.shape[3], inputs.shape[4])
        
    @pytest.mark.parametrize("inputs, embed_dim, num_heads", [
        (torch.rand(2, 8, 128), 128, 8),
        (torch.rand(2, 64, 256), 256, 8),
        (torch.rand(2, 256, 96), 96, 8),
    ])
    def test_multi_head_attention(self, inputs, embed_dim, num_heads):
        op = MultiHeadAttention(embed_size=embed_dim, num_heads=num_heads)
        output = op(inputs)
        assert output.shape == inputs.shape
        
    @pytest.mark.parametrize("inputs", [
        torch.rand(2, 8, 128),
        torch.rand(4, 16, 32),
        torch.rand(8, 32, 64),
    ])
    def test_mlp(self, inputs):
        op = MLP(embed_size=inputs.shape[2])
        output = op(inputs)
        assert output.shape == inputs.shape
        
    @pytest.mark.parametrize("inputs", [
        (1, ),
        (2, ),
        (torch.tensor([10]))            
    ])
    def test_residual_add(self, inputs):
        op = ResidualAdd(lambda x: x)
        output = op(inputs)
        assert output == 2 * inputs

    @pytest.mark.parametrize("inputs, transform", [
        (torch.rand(2, 8, 128), torch.nn.Conv1d(8, 8, 1)),
        (torch.rand(2, 12, 128), lambda x: x),
        (torch.rand(2, 16, 128), torch.nn.Linear(128, 128)),
    ])
    def test_pre_norm(self, inputs, transform):
        op = PreNorm(block=transform, dim=inputs.shape[2])
        output = op(inputs)
        assert output.shape == inputs.shape
