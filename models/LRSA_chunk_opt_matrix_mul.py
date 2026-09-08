import torch
import torch.nn as nn

from perceiverforpde.modeling.perceiver import (
    PerceiverModel,
    PerceiverModelConfig,
)

from perceiverforpde.modeling.layers.attn import (
    PerceiverAttentionConfig,
)

from perceiverforpde.modeling.layers.embed_position import (
    EmbedPositionConfig,
)

from perceiverforpde.modeling.layers.mlp import (
    FeedForwardWithGatingConfig,
    MultiLayerFeedForwardConfig,
)


def _as_feature_tensor(chunk):
    if isinstance(chunk, (tuple, list)):
        return chunk[0]
    return chunk


class Model(nn.Module):
    """
    LRSA wrapper for the existing AhmedML / Transolver-3 pipeline.

    AhmedML input:
        [..., 0:3] = xyz coordinates
        [..., 3:6] = surface normals

    Output:
        [..., 0]   = pressure
        [..., 1:4] = wall shear stress vector
    """

    def __init__(
        self,
        space_dim=6,
        n_layers=8,
        n_hidden=256,
        dropout=0.0,
        n_head=8,
        act="gelu",
        mlp_ratio=1,
        fun_dim=0,
        out_dim=4,
        slice_num=64,
        ref=8,
        unified_pos=False,
        dim_head=None,
        **kwargs,
    ):
        super().__init__()

        if dim_head is None:
            if n_hidden % n_head != 0:
                raise ValueError(
                    f"n_hidden={n_hidden} must be divisible by n_head={n_head}"
                )
            dim_head = n_hidden // n_head

        if space_dim != 6:
            raise ValueError(
                f"AhmedML LRSA wrapper expects space_dim=6, got {space_dim}"
            )

        # Official ShapeNetCar-style LRSA configuration.
        embed_position = EmbedPositionConfig(
            name="sinusoidal",
            num_freqs=9,
            min_freqs_exp=-4,
            max_freqs_exp=4,
            include_input=True,
        )

        lifting = MultiLayerFeedForwardConfig(
            hidden_features=256,
            num_hidden_layers=0,
            bias=True,
            act="gelu",
        )

        project = MultiLayerFeedForwardConfig(
            hidden_features=128,
            num_hidden_layers=0,
            bias=True,
            act="gelu",
        )

        latent_ffn = FeedForwardWithGatingConfig(
            hidden_features=None,
            mlp_ratio=1,
            bias=True,
            act="gelu",
            disable_gate=True,
            dropout=0.0,
        )

        point_ffn = FeedForwardWithGatingConfig(
            hidden_features=None,
            mlp_ratio=1,
            bias=True,
            act="gelu",
            disable_gate=True,
            dropout=0.0,
        )

        attn = PerceiverAttentionConfig(
            num_heads=n_head,
            num_latents=slice_num,
            bias=True,
            ffn=latent_ffn,
            dim_heads=dim_head,

            enable_rope=False,

            qk_norm=True,
            norm_type="rmsnorm",

            disable_interleaved_blocks=False,
            disable_interleaved_channel_mixing=False,

            enable_gated_attention_up=False,

            attention_bias_up_q=False,
            attention_bias_interleaved_q=False,
        )

        config = PerceiverModelConfig(
            d_model=n_hidden,
            num_layers=n_layers,

            embed_position=embed_position,
            lifting=lifting,
            project=project,

            attn=attn,
            ffn=point_ffn,

            enable_out_proj_norm=False,
            norm_type="rmsnorm",
            init_type="mitchell",
        )

        # xyz = physical coordinates
        # normals = functional features
        self.model = PerceiverModel(
            phys_dim=3,
            func_dim=3,
            out_dim=out_dim,
            config=config,
        )

        self.__name__ = "LRSA_AhmedML"

    def _forward_tensor(self, x):
        xyz = x[..., :3]
        normals = x[..., 3:6]

        return self.model(
            xyz,
            normals,
        )

    def forward(
        self,
        data,
        use_checkpoint=True,
        input_list=True,
    ):
        if not input_list:
            x = _as_feature_tensor(data)
            return self._forward_tensor(x)

        chunks = [
            _as_feature_tensor(c)
            for c in data
        ]

        # If only one chunk is supplied, avoid unnecessary concatenate/split.
        if len(chunks) == 1:
            return [
                self._forward_tensor(chunks[0])
            ]

        # Treat all supplied chunks as one point cloud so LRSA's
        # low-rank compression is global across them.
        sizes = [
            c.shape[1]
            for c in chunks
        ]

        x = torch.cat(
            chunks,
            dim=1,
        )

        out = self._forward_tensor(x)

        return list(
            torch.split(
                out,
                sizes,
                dim=1,
            )
        )
