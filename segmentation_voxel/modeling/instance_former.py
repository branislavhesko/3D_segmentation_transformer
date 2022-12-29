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
        feats = einops.rearrange(feats, "b c n-> b n c")
        return self.linear(feats)


class DeconvLayer(nn.Sequential):

    def __init__(self, in_features, out_features, kernel_size, stride=2, use_conv=True) -> None:
        super().__init__()
        self.layers = [nn.ConvTranspose3d(in_features, out_features, kernel_size, stride)]
        if use_conv:
            self.layers.append(ConvBatchNormRelu(out_features, out_features, kernel_size=3, stride=1, padding=1))
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


class Decoder(nn.Module):
    # TODO: make more general
    FEATURE_NUMBERS = {
        "small": [96, 64, 48, 32, 16],
        "medium": [256, 192, 128, 64, 32],
        "large": [768, 512, 256, 128, 64]
    }

    def __init__(self, num_deconv_blocks, model_size: str = "large") -> None:
        super().__init__()
        self.feature_numbers = self.FEATURE_NUMBERS[model_size]
        self.deconv_blocks = nn.ModuleList([
                DeconvLayer(self.feature_numbers[idx], self.feature_numbers[idx + 1], kernel_size=2, stride=2, use_conv=True)
                for idx in range(num_deconv_blocks)
        ])
        self.conv_blocks = nn.Sequential(*[
            ConvBatchNormRelu(self.feature_numbers[num_deconv_blocks] * 2, self.feature_numbers[num_deconv_blocks], padding=1),
            ConvBatchNormRelu(self.feature_numbers[num_deconv_blocks], self.feature_numbers[num_deconv_blocks + 1], padding=1),
        ])
        self.final_deconv = nn.ConvTranspose3d(self.feature_numbers[num_deconv_blocks + 1], self.feature_numbers[num_deconv_blocks + 1], 2, 2)

    def forward(self, features, features_lower):
        for block in self.deconv_blocks:
            features = block(features)
        features = torch.cat([features, features_lower], dim=1)
        features = self.conv_blocks(features)
        features = self.final_deconv(features)
        return features


class TransformerDecoder(nn.Module):

    def __init__(self, num_layers, num_queries, num_heads, hidden_dim, output_classes) -> None:
        super().__init__()
        self.queries = nn.Embedding(num_queries, hidden_dim)
        self.decoder = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
            for _ in range(num_layers)]
        )
        self.classification_layer = nn.Linear(hidden_dim, output_classes)

    def forward(self, features):
        queries = self.queries.weight.unsqueeze(0).repeat(features["layer4"].shape[0], 1, 1)
        for decoder, key in zip(self.decoder, features.keys()):
            queries = decoder(queries, einops.rearrange(features[key], "b c h w d -> b (h w d) c"))
        return self.classification_layer(queries.mean(dim=1))


class InstanceSegmentationFormer3D(nn.Module):
    direct_block_channels = {"small": 16, "medium": 32, "large": 64}
    extraction_layers = {3: "layer1", 6: "layer2", 9: "layer3", 12: "layer4"}
    last_conv_channels = {
        "small": 64,
        "medium": 192,
        "large": 512
    }
    final_conv_channels = {
        "small": (32, 16),
        "medium": (64, 32),
        "large": (128, 64)
    }

    def __init__(
            self,
            num_classes,
            num_queries,
            embed_size,
            num_heads,
            input_channels,
            channels,
            patch_size,
            input_shape,
            dropout
    ) -> None:
        super().__init__()
        assert embed_size in [768, 256, 96]
        self.model_size = "large" if embed_size == 768 else "medium" if embed_size == 256 else "small"
        self.encoder = nn.ModuleList([TransformerEncoderLayer(embed_size, num_heads) for _ in range(12)])
        num_embeddings = (input_shape[0] // patch_size) * (input_shape[1] // patch_size) * (input_shape[2] // patch_size)
        self.positional_encoding = nn.Parameter(torch.randn(1, num_embeddings, embed_size))
        self.embedding = VoxelEmbedding(input_channels, embed_size, stride=patch_size)
        self.direct_block = nn.Sequential(
            *[ConvBatchNormRelu(input_channels, self.direct_block_channels[self.model_size], kernel_size=3, stride=1, padding=1),
              ConvBatchNormRelu(
                  self.direct_block_channels[self.model_size],
                  self.direct_block_channels[self.model_size],
                  kernel_size=3, stride=1, padding=1)])
        self.decoder3 = Decoder(1, model_size=self.model_size)
        self.decoder2 = Decoder(2, model_size=self.model_size)
        self.decoder1 = Decoder(3, model_size=self.model_size)
        self.patch_size = patch_size
        self.last_deconv = nn.ConvTranspose3d(embed_size, self.last_conv_channels[self.model_size], 2, 2)
        self.final_layer = nn.Sequential(
            *[
                ConvBatchNormRelu(self.final_conv_channels[self.model_size][0], self.final_conv_channels[self.model_size][1], kernel_size=3, stride=1, padding=1),
                ConvBatchNormRelu(self.final_conv_channels[self.model_size][1], self.final_conv_channels[self.model_size][1], kernel_size=3, stride=1, padding=1),
                nn.Conv3d(self.final_conv_channels[self.model_size][1], num_queries, kernel_size=1, stride=1, padding=0),
            ]
        )
        self.transformer_decoder = TransformerDecoder(4, num_queries, num_heads, embed_size, num_classes)

    def forward(self, volume):
        b, c, h, w, d = volume.shape
        embedding = self.embedding(volume)

        features = {}
        for encoder_index, encoder in enumerate(self.encoder):
            embedding = encoder(embedding + self.positional_encoding)
            if encoder_index + 1 in self.extraction_layers:
                 features[self.extraction_layers[encoder_index + 1]] = einops.rearrange(
                     embedding, "b (h w d) e -> b e h w d",
                     h=h // self.patch_size,
                     w=w // self.patch_size,
                     d=d // self.patch_size
                )

        processed_direct_features = self.direct_block(volume)
        decoder3_out = self.decoder3(features["layer3"], self.last_deconv(features["layer4"]))
        decoder2_out = self.decoder2(features["layer2"], decoder3_out)
        decoder1_out = self.decoder1(features["layer1"], decoder2_out)

        decoder_classes = self.transformer_decoder(features)

        final_features = torch.cat([processed_direct_features, decoder1_out], dim=1)
        return self.final_layer(final_features), decoder_classes


if __name__ == "__main__":
    model = InstanceSegmentationFormer3D(
        2, 50, 256, 8, 1, None, 16, (64, 64, 64), 0.1
    )

    with torch.no_grad():
        print(model(torch.rand(20, 1, 64, 64, 64))[0].shape)
