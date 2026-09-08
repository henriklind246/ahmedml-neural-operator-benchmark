import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange


ACTIVATION = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
}

trunc_normal_ = nn.init.trunc_normal_


def _as_feature_tensor(chunk):
    if isinstance(chunk, (tuple, list)):
        return chunk[0]
    return chunk


# ============================================================
# LinearNO attention
# Based on the official HiPRL/LinearNO implementation.
#
# Standard form:
#
#   Q = softmax(Q, feature dimension)
#   K = softmax(K, spatial/point dimension)
#
#   output = Q (K^T V)
#
# The chunked implementation below computes K^T V globally
# across every supplied chunk.
# ============================================================

class LinearNOAttention(nn.Module):

    def __init__(
        self,
        dim,
        heads=8,
        dim_head=32,
        dropout=0.0,
        key_ratio=4,
    ):
        super().__init__()

        inner_dim = heads * dim_head

        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.key_ratio = key_ratio
        self.key_dim = key_ratio * dim_head

        self.dropout = nn.Dropout(dropout)

        # Preserve spelling/initialization behavior of official LinearNO.
        self.temperature_q = nn.Parameter(
            torch.ones(1, heads, 1, 1) * 0.5
        )

        self.temperature_k = nn.Parameter(
            torch.ones(1, heads, 1, 1) * 0.5
        )

        self.in_project_x = nn.Linear(dim, inner_dim)

        self.to_q = nn.Linear(
            dim_head,
            self.key_dim,
            bias=False,
        )

        self.to_k = nn.Linear(
            dim_head,
            self.key_dim,
            bias=False,
        )

        self.to_v = nn.Linear(
            dim_head,
            dim_head,
            bias=False,
        )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

    def _project(self, x):
        """
        x: (B, N, C)

        Returns
        -------
        x_mid : (B, H, N, D)
        q_raw : (B, H, N, key_dim)
        k_raw : (B, H, N, key_dim)
        v     : (B, H, N, D)
        """

        x_mid = self.in_project_x(x)

        x_mid = rearrange(
            x_mid,
            "b n (h d) -> b h n d",
            h=self.heads,
            d=self.dim_head,
        )

        q_raw = self.to_q(x_mid)
        k_raw = self.to_k(x_mid)
        v = self.to_v(x_mid)

        return x_mid, q_raw, k_raw, v

    # --------------------------------------------------------
    # Ordinary, single-tensor LinearNO attention
    # --------------------------------------------------------

    def forward(self, x):

        _, q, k, v = self._project(x)

        tq = torch.clamp(
            self.temperature_q,
            min=0.1,
            max=2.0,
        )

        tk = torch.clamp(
            self.temperature_k,
            min=0.1,
            max=2.0,
        )

        # Official LinearNO normalization.
        q = F.softmax(q / tq, dim=-1)
        k = F.softmax(k / tk, dim=-2)

        # K^T V
        kv = torch.einsum(
            "bhnd,bhnc->bhdc",
            k,
            v,
        )

        # Q(K^T V)
        qkv = torch.einsum(
            "bhnd,bhdc->bhnc",
            q,
            kv,
        )

        qkv = rearrange(
            qkv,
            "b h n d -> b n (h d)",
        )

        return self.to_out(qkv)

    # --------------------------------------------------------
    # Chunked / large-mesh support
    # --------------------------------------------------------

    def chunk_kv_stats(self, x):
        """
        Compute sufficient statistics for global K^T V.

        Important:
        LinearNO normalizes K across ALL spatial points.
        Therefore simply computing an independent softmax
        inside each chunk would be incorrect.

        We return numerically stable local statistics that
        can later be combined across chunks.

        Returns
        -------
        local_max : (B,H,key_dim)
        local_den : (B,H,key_dim)
        local_num : (B,H,key_dim,D)
        """

        _, _, k_raw, v = self._project(x)

        tk = torch.clamp(
            self.temperature_k,
            min=0.1,
            max=2.0,
        )

        logits = k_raw / tk

        # Stable softmax statistics over local points.
        local_max = logits.amax(
            dim=2
        )

        exp_logits = torch.exp(
            logits - local_max.unsqueeze(2)
        )

        local_den = exp_logits.sum(
            dim=2
        )

        local_num = torch.einsum(
            "bhnd,bhnc->bhdc",
            exp_logits,
            v,
        )

        return local_max, local_den, local_num

    @staticmethod
    def merge_kv_stats(
        global_max,
        global_den,
        global_num,
        local_max,
        local_den,
        local_num,
    ):
        """
        Stable merge of softmax sufficient statistics.
        """

        if global_max is None:
            return local_max, local_den, local_num

        new_max = torch.maximum(
            global_max,
            local_max,
        )

        old_scale = torch.exp(
            global_max - new_max
        )

        new_scale = torch.exp(
            local_max - new_max
        )

        global_den = (
            global_den * old_scale
            + local_den * new_scale
        )

        global_num = (
            global_num * old_scale.unsqueeze(-1)
            + local_num * new_scale.unsqueeze(-1)
        )

        return new_max, global_den, global_num

    def apply_global_kv(self, x, kv):
        """
        Apply Q(K^T V) to one chunk using the global K^T V.
        """

        _, q, _, _ = self._project(x)

        tq = torch.clamp(
            self.temperature_q,
            min=0.1,
            max=2.0,
        )

        q = F.softmax(
            q / tq,
            dim=-1,
        )

        qkv = torch.einsum(
            "bhnd,bhdc->bhnc",
            q,
            kv,
        )

        qkv = rearrange(
            qkv,
            "b h n d -> b n (h d)",
        )

        return self.to_out(qkv)


# ============================================================
# MLP
# ============================================================

class MLP(nn.Module):

    def __init__(
        self,
        n_input,
        n_hidden,
        n_output,
        n_layers=1,
        act="gelu",
        res=True,
    ):
        super().__init__()

        if act not in ACTIVATION:
            raise NotImplementedError(act)

        act_cls = ACTIVATION[act]

        self.n_layers = n_layers
        self.res = res

        self.linear_pre = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            act_cls(),
        )

        self.linear_post = nn.Linear(
            n_hidden,
            n_output,
        )

        self.linears = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_hidden, n_hidden),
                act_cls(),
            )
            for _ in range(n_layers)
        ])

    def forward(self, x):

        x = self.linear_pre(x)

        for layer in self.linears:
            if self.res:
                x = layer(x) + x
            else:
                x = layer(x)

        return self.linear_post(x)


# ============================================================
# LinearNO block
# ============================================================

class LinearNOBlock(nn.Module):

    def __init__(
        self,
        num_heads,
        hidden_dim,
        dropout,
        act="gelu",
        mlp_ratio=2,
        key_ratio=4,
        last_layer=False,
        out_dim=4,
    ):
        super().__init__()

        self.last_layer = last_layer

        self.ln_1 = nn.LayerNorm(
            hidden_dim
        )

        self.Attn = LinearNOAttention(
            hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            key_ratio=key_ratio,
        )

        self.ln_2 = nn.LayerNorm(
            hidden_dim
        )

        self.mlp = MLP(
            hidden_dim,
            hidden_dim * mlp_ratio,
            hidden_dim,
            n_layers=0,
            res=False,
            act=act,
        )

        if self.last_layer:
            self.ln_3 = nn.LayerNorm(
                hidden_dim
            )

            self.mlp2 = nn.Linear(
                hidden_dim,
                out_dim,
            )

    def forward(self, fx):

        fx = (
            self.Attn(
                self.ln_1(fx)
            )
            + fx
        )

        fx = (
            self.mlp(
                self.ln_2(fx)
            )
            + fx
        )

        if self.last_layer:
            fx = self.mlp2(
                self.ln_3(fx)
            )

        return fx

    def forward_chunks(
        self,
        fx_list,
        eps=1e-12,
        use_checkpoint=True,
    ):
        """
        Exact chunked LinearNO attention.

        Pass 1:
            accumulate global softmax-normalized K^T V

        Pass 2:
            apply each chunk's Q to the same global K^T V
        """

        global_max = None
        global_den = None
        global_num = None

        # ------------------------
        # Pass 1: global K^T V
        # ------------------------

        for fxk in fx_list:

            uk = self.ln_1(fxk)

            if use_checkpoint:
                local_max, local_den, local_num = checkpoint(
                    self.Attn.chunk_kv_stats,
                    uk,
                    use_reentrant=False,
                )
            else:
                local_max, local_den, local_num = (
                    self.Attn.chunk_kv_stats(uk)
                )

            (
                global_max,
                global_den,
                global_num,
            ) = self.Attn.merge_kv_stats(
                global_max,
                global_den,
                global_num,
                local_max,
                local_den,
                local_num,
            )

        # This is the globally normalized K^T V.
        kv = global_num / (
            global_den.unsqueeze(-1) + eps
        )

        # ------------------------
        # Pass 2: Q(K^T V)
        # ------------------------

        out_list = []

        for fxk in fx_list:

            def chunk_compute(f_k, global_kv):

                uk = self.ln_1(f_k)

                attn_out = (
                    self.Attn.apply_global_kv(
                        uk,
                        global_kv,
                    )
                )

                res = attn_out + f_k

                res = (
                    self.mlp(
                        self.ln_2(res)
                    )
                    + res
                )

                if self.last_layer:
                    res = self.mlp2(
                        self.ln_3(res)
                    )

                return res

            if use_checkpoint:
                fxk_new = checkpoint(
                    chunk_compute,
                    fxk,
                    kv,
                    use_reentrant=False,
                )
            else:
                fxk_new = chunk_compute(
                    fxk,
                    kv,
                )

            out_list.append(
                fxk_new
            )

        return out_list


# ============================================================
# Model
# ============================================================

class Model(nn.Module):

    def __init__(
        self,
        space_dim=1,
        n_layers=5,
        n_hidden=256,
        dropout=0.0,
        n_head=8,
        act="gelu",
        mlp_ratio=2,
        fun_dim=1,
        out_dim=1,
        key_ratio=4,

        # Accepted for compatibility with Transolver-3's
        # MODEL_KWARGS. LinearNO does not use physical slices.
        slice_num=64,

        ref=8,
        unified_pos=False,
    ):
        super().__init__()

        self.__name__ = "LinearNO_AhmedML"

        self.ref = ref
        self.unified_pos = unified_pos
        self.slice_num = slice_num
        self.key_ratio = key_ratio

        if self.unified_pos:
            raise NotImplementedError(
                "unified_pos is not needed for AhmedML"
            )

        self.preprocess = MLP(
            fun_dim + space_dim,
            n_hidden * 2,
            n_hidden,
            n_layers=0,
            res=False,
            act=act,
        )

        self.blocks = nn.ModuleList([
            LinearNOBlock(
                num_heads=n_head,
                hidden_dim=n_hidden,
                dropout=dropout,
                act=act,
                mlp_ratio=mlp_ratio,
                key_ratio=key_ratio,
                out_dim=out_dim,
                last_layer=(i == n_layers - 1),
            )
            for i in range(n_layers)
        ])

        self.placeholder = nn.Parameter(
            (1.0 / n_hidden)
            * torch.rand(
                n_hidden,
                dtype=torch.float,
            )
        )

        self.initialize_weights()

    def initialize_weights(self):

        self.apply(
            self._init_weights
        )

    @staticmethod
    def _init_weights(m):

        if isinstance(m, nn.Linear):

            trunc_normal_(
                m.weight,
                std=0.02,
            )

            if m.bias is not None:
                nn.init.constant_(
                    m.bias,
                    0,
                )

        elif isinstance(
            m,
            (
                nn.LayerNorm,
                nn.BatchNorm1d,
            ),
        ):

            nn.init.constant_(
                m.bias,
                0,
            )

            nn.init.constant_(
                m.weight,
                1.0,
            )

    def forward(
        self,
        data,
        use_checkpoint=True,
        input_list=True,
    ):

        if input_list:

            fx_list = []

            for chunk in data:

                x = _as_feature_tensor(
                    chunk
                )

                fx = self.preprocess(
                    x
                )

                fx = (
                    fx
                    + self.placeholder[
                        None,
                        None,
                        :
                    ]
                )

                fx_list.append(
                    fx
                )

            for block in self.blocks:

                fx_list = (
                    block.forward_chunks(
                        fx_list,
                        use_checkpoint=use_checkpoint,
                    )
                )

            return fx_list

        x = _as_feature_tensor(
            data
        )

        fx = self.preprocess(
            x
        )

        fx = (
            fx
            + self.placeholder[
                None,
                None,
                :
            ]
        )

        for block in self.blocks:
            fx = block(fx)

        return fx
