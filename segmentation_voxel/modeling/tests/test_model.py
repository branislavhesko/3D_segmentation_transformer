import pytest
from segmentation_voxel.modeling.voxel_former import (
    ConvBatchNormRelu,
    Decoder, 
    DeconvLayer, 
    PreNorm, 
    MLP, 
    MultiHeadAttention, 
    ResidualAdd, 
    TransformerEncoderLayer, 
    VoxelEmbedding, 
    SegmentationTransformer3D
)
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

    @pytest.mark.parametrize("inputs, embed_dim, num_heads, dropout_probability", [
        (torch.rand(2, 197, 96), 96, 8, 0.),
        (torch.rand(2, 7, 96), 96, 8, 0.5),
        (torch.rand(2, 197, 128), 128, 8, 0.2),
    ])
    def test_transformer_encoder_layer(self, inputs, embed_dim, num_heads, dropout_probability):
        layer = TransformerEncoderLayer(embed_dim, num_heads, dropout_probability)
        out = layer(inputs)
        assert out.shape == inputs.shape

    @pytest.mark.parametrize("inputs, stride, kernel_size", [
        (torch.rand(2, 3, 16, 16, 16), 2, 2)
    ])
    def test_deconvolution(self, inputs, stride, kernel_size):
        op = DeconvLayer(inputs.shape[1], inputs.shape[1], stride=stride, kernel_size=kernel_size)
        out = op(inputs)
        assert out.shape == (inputs.shape[0], inputs.shape[1], inputs.shape[2] * stride, inputs.shape[3] * stride, inputs.shape[4] * stride)

    @pytest.mark.parametrize("inputs, num_classes, num_heads, embed_dim, device", [
        (torch.rand(2, 3, 64, 64, 64), 2, 8, 768, "cpu"),
        # (torch.rand(2, 3, 64, 64, 64), 2, 16, 768, "cuda"),
        (torch.rand(2, 4, 32, 32, 32), 2, 4, 768, "cpu"),
        # (torch.rand(2, 4, 96, 96, 96), 2, 8, 96, "cuda"),
        (torch.rand(2, 3, 128, 128, 128), 2, 16, 96, "cpu"),
        # (torch.rand(2, 3, 64, 64, 64), 2, 4, 256, "cuda"),
        (torch.rand(2, 3, 64, 64, 64), 2, 2, 256, "cpu")
    ])
    def test_segmentation_transformer_3d_inference(self, inputs, num_classes, num_heads, embed_dim, device):
        patch_size = 16
        num_patches = inputs.shape[2] * inputs.shape[3] * inputs.shape[4] // (patch_size ** 3)
        out = SegmentationTransformer3D(
            num_classes=num_classes, 
            embed_size=embed_dim, 
            num_heads=num_heads, 
            input_channels=inputs.shape[1], 
            channels=8, 
            patch_size=patch_size,
            input_shape=num_patches, 
            dropout=0.1).to(device)(inputs.to(device))
        assert isinstance(out, torch.Tensor)
        assert out.shape == (inputs.shape[0], 2, inputs.shape[2], inputs.shape[3], inputs.shape[4])

    @pytest.mark.parametrize("features, features_lower, num_deconv_blocks", [
        (torch.rand(2, 768, 8, 8, 8), torch.rand(2, 512, 16, 16, 16), 1),
        (torch.rand(2, 768, 8, 8, 8), torch.rand(2, 256, 32, 32, 32), 2),
        (torch.rand(2, 768, 8, 8, 8), torch.rand(2, 128, 64, 64, 64), 3)

    ])
    def test_decoder(self, features, features_lower, num_deconv_blocks):
        decoder = Decoder(num_deconv_blocks)
        out = decoder(features, features_lower)
        print(out)
