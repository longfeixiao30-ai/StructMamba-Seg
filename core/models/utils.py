import torch
import torch.nn as nn
import math
from functools import partial
from einops import rearrange, repeat
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import einops
from .LPRC_class_CE import LPRCGradCompensator

try:
    "sscore acts the same as mamba_ssm"
    SSMODE = "sscore"
    import selective_scan_cuda_core
except Exception as e:
    print(e, flush=True)
    "you should install mamba_ssm to use this"
    SSMODE = "mamba_ssm"
    import selective_scan_cuda
    # from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref


def weak_init_conv_offset_modules(module: nn.Module):
    """
    对 segmentation-oriented offset predictor 做“弱影响初始化”：
    - norm: identity
    - 普通 conv: 小尺度初始化
    - 最终 pred / fuse / edge_head 最后层: 置零，保证初始 offset 很小
    """
    for name, m in module.named_modules():
        # 1) norm 类 -> 尽量恒等
        if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            if getattr(m, "weight", None) is not None:
                nn.init.ones_(m.weight)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

        # 2) conv -> 小尺度初始化
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            m.weight.data.mul_(0.1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        # 3) 关键输出层置零：让一开始几乎不偏移
        if isinstance(m, nn.Conv2d):
            if (
                    name.endswith("pred")
                    or name.endswith("edge_head.2")
                    or name.endswith("fuse.3")
                    or name.endswith("ctx_weight.3")
            ):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class SelectiveScan(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1):
        assert nrows in [1, 2, 3, 4], f"{nrows}"  # 8+ is too slow to compile
        assert u.shape[1] % (B.shape[1] * nrows) == 0, f"{nrows}, {u.shape}, {B.shape}"
        ctx.delta_softplus = delta_softplus
        ctx.nrows = nrows
        # all in float
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None and D.stride(-1) != 1:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if B.dim() == 3:
            B = B.unsqueeze(dim=1)
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = C.unsqueeze(dim=1)
            ctx.squeeze_C = True

        if SSMODE == "mamba_ssm":
            out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, delta_softplus)
        else:
            out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()

        if SSMODE == "mamba_ssm":
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                u, delta, A, B, C, D, None, delta_bias, dout, x, None, None, ctx.delta_softplus,
                False  # option to recompute out_z, not used here
            )
        else:
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
                u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
                # u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, ctx.nrows,
            )

        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None)


class DeformablePathTrans(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, de_index):
        B, C, N = x.shape
        _, indices = torch.topk(de_index, k=N, dim=-1, largest=False)
        x_gathered = torch.gather(x, 2, indices.unsqueeze(1).expand(-1, C, -1)).contiguous()
        x_out = x_gathered.permute(0, 2, 1).contiguous()
        ctx.save_for_backward(x, de_index, indices)
        return x_out, indices

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        x, de_index, indices = ctx.saved_tensors
        grad_x = torch.zeros_like(x)
        grad_x.scatter_add_(2, indices.unsqueeze(1).expand(-1, x.shape[1], -1),
                            grad_output.permute(0, 2, 1).contiguous()).contiguous()
        grad_de_index = (grad_output.permute(0, 2, 1).contiguous() - grad_x).mean(dim=1)
        grad_de_index = grad_de_index.view_as(de_index)
        return grad_x, grad_de_index


class ConvOffset(nn.Module):
    """
    Main Innovation 1 - Boundary-Context Guided Offset Predictor
    with explicit ablation switches.

    Switches:
    use_ep: Edge Prompt branch
    use_mc: Multi-scale Context branch
    use_sa: Stage-Aware behavior
        - controls local kernel size
        - controls dp/dt/edge/context stage-aware scales

    Notes:
    1) Local branch is always kept as the minimal predictor backbone.
    2) When EP/MC are disabled, we feed zero tensors into the fusion stage
       to keep tensor shapes unchanged.
    3) When SA is disabled, we use a unified kernel size and unified scales,
       so the predictor becomes stage-agnostic.
    """

    def __init__(
            self,
            embed_dim,
            stage=0,
            debug=False,
            use_ep=True,
            use_mc=True,
            use_sa=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.stage = stage
        self.debug = debug

        # ---------------------------------------------------------
        # Ablation switches
        # ---------------------------------------------------------
        self.use_ep = use_ep
        self.use_mc = use_mc
        self.use_sa = use_sa

        # ---------------------------------------------------------
        # Stage-aware local kernel size
        # - SA on : use different receptive fields across stages
        # - SA off: use one unified kernel size for all stages
        # ---------------------------------------------------------
        if self.use_sa:
            self.local_ks = [9, 7, 5, 3][stage]
        else:
            self.local_ks = 3

        self.local_pad = self.local_ks // 2
        mid_dim = max(embed_dim // 4, 8)

        # =========================================================
        # 1) Local feature branch (always kept)
        # =========================================================
        self.local_dw = nn.Conv2d(
            embed_dim, embed_dim,
            kernel_size=self.local_ks,
            padding=self.local_pad,
            groups=embed_dim,
            bias=False
        )

        # =========================================================
        # 2) Edge Prompt Branch (EP)
        # =========================================================
        self.edge_dw = nn.Conv2d(
            embed_dim, embed_dim,
            kernel_size=3,
            padding=1,
            groups=embed_dim,
            bias=False
        )
        self.edge_head = nn.Sequential(
            nn.Conv2d(embed_dim, mid_dim, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(mid_dim, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        # =========================================================
        # 3) Multi-scale Context Branch (MC)
        # =========================================================
        self.ctx_conv1 = nn.Conv2d(
            embed_dim, embed_dim,
            kernel_size=3, padding=1,
            groups=embed_dim, bias=False
        )
        self.ctx_conv2 = nn.Conv2d(
            embed_dim, embed_dim,
            kernel_size=3, padding=2, dilation=2,
            groups=embed_dim, bias=False
        )
        self.ctx_conv3 = nn.Conv2d(
            embed_dim, embed_dim,
            kernel_size=3, padding=3, dilation=3,
            groups=embed_dim, bias=False
        )
        self.ctx_weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embed_dim, mid_dim, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(mid_dim, 3, kernel_size=1, bias=True)
        )

        # =========================================================
        # 4) Boundary-Context Joint Fusion
        # Input channels stay fixed:
        # [local_feat, ctx_feat, edge_map]
        # =========================================================
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 2 + 1, embed_dim, kernel_size=1, bias=False),
            nn.GroupNorm(1, embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False),
            nn.GELU()
        )

        # =========================================================
        # 5) Final predictor
        # Output: [dx, dy, dt, g]
        # =========================================================
        self.pred = nn.Conv2d(embed_dim, 4, kernel_size=1, bias=True)

        # =========================================================
        # 6) Stage-aware DP/DT emphasis
        # - SA on : stage-dependent scales
        # - SA off: unified scales
        # =========================================================
        if self.use_sa:
            if stage < 2:
                self.dp_scale = 1.0
                self.dt_scale = 0.25
                self.edge_scale = 1.0
                self.ctx_scale = 0.80
            else:
                self.dp_scale = 0.45
                self.dt_scale = 1.0
                self.edge_scale = 0.55
                self.ctx_scale = 1.00
        else:
            self.dp_scale = 1.0
            self.dt_scale = 1.0
            self.edge_scale = 1.0
            self.ctx_scale = 1.0

        weak_init_conv_offset_modules(self)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            pred: [B, 4, H, W]
            pred[:, :2] -> spatial offsets Δp
            pred[:, 2:3] -> path/index offset Δt
            pred[:, 3:4] -> adaptive gate g
        """
        # =========================================================
        # Local structure feature (always kept)
        # =========================================================
        local_feat = self.local_dw(x)

        # =========================================================
        # Edge Prompt Branch (EP)
        # =========================================================
        if self.use_ep:
            edge_feat = self.edge_dw(x)
            edge_map = self.edge_head(edge_feat) * self.edge_scale  # [B, 1, H, W]
        else:
            edge_map = torch.zeros(
                x.size(0), 1, x.size(2), x.size(3),
                device=x.device, dtype=x.dtype
            )

        # =========================================================
        # Multi-scale Context Branch (MC)
        # =========================================================
        if self.use_mc:
            c1 = self.ctx_conv1(x)
            c2 = self.ctx_conv2(x)
            c3 = self.ctx_conv3(x)
            ctx_w = self.ctx_weight(x)  # [B, 3, 1, 1]
            ctx_w = torch.softmax(ctx_w, dim=1)

            ctx_feat = self.ctx_scale * (
                    ctx_w[:, 0:1] * c1 +
                    ctx_w[:, 1:2] * c2 +
                    ctx_w[:, 2:3] * c3
            )
        else:
            ctx_feat = torch.zeros_like(local_feat)

        # =========================================================
        # Boundary-Context Guided Fusion
        # =========================================================
        fused = torch.cat([local_feat, ctx_feat, edge_map], dim=1)
        fused = self.fuse(fused)

        # =========================================================
        # Predict [dx, dy, dt, g]
        # =========================================================
        pred = self.pred(fused)
        pred_dp = pred[:, 0:2] * self.dp_scale
        pred_dt = pred[:, 2:3] * self.dt_scale
        gate = torch.sigmoid(pred[:, 3:4])
        pred = torch.cat([pred_dp, pred_dt, gate], dim=1)

        if self.debug:
            print(
                f"[ConvOffset] stage={self.stage}, "
                f"use_ep={self.use_ep}, use_mc={self.use_mc}, use_sa={self.use_sa}, "
                f"dp_scale={self.dp_scale}, dt_scale={self.dt_scale}"
            )

        return pred


class DeformableLayer(nn.Module):
    """
    Deformable scanning layer with ablation switches.

    Switches:
    use_ep: passed to ConvOffset
    use_mc: passed to ConvOffset
    use_pc: whether to enable position compensation
    use_sa: passed to ConvOffset
    """

    def __init__(
            self,
            index=0,
            embed_dim=192,
            debug=False,
            h=0,
            w=0,
            use_ep=True,
            use_mc=True,
            use_pc=True,
            use_sa=True,
    ):
        super().__init__()
        self.debug = debug

        # ---------------------------------------------------------
        # Ablation switches
        # ---------------------------------------------------------
        self.use_ep = use_ep
        self.use_mc = use_mc
        self.use_pc = use_pc
        self.use_sa = use_sa

        # ---------------------------------------------------------
        # Upgraded offset predictor with component switches
        # ---------------------------------------------------------
        self.conv_offset = ConvOffset(
            embed_dim,
            stage=index,
            debug=debug,
            use_ep=use_ep,
            use_mc=use_mc,
            use_sa=use_sa,
        )

        self.rpe_table = nn.Parameter(
            torch.zeros(embed_dim, 7, 7)
        )
        trunc_normal_(self.rpe_table, std=0.01)

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H_key - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B, -1, -1, -1)  # B H W 2
        return ref

    @torch.no_grad()
    def _get_key_ref_points(self, H, W, B, dtype, device):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0, H, H, dtype=dtype, device=device),
            torch.linspace(0, W, W, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B, -1, -1, -1)  # B H W 2
        return ref

    @torch.no_grad()
    def _get_path_ref_points(self, N, B, dtype, device):
        ref_path = torch.linspace(0.5, N - 0.5, N, dtype=dtype, device=device)
        ref_path.div_(N - 1.0).mul_(2.0).sub_(1.0)
        ref = ref_path[None, ...].expand(B, -1)
        return ref

    @torch.no_grad()
    def _get_path_key_ref_points(self, N, B, dtype, device):
        ref_path = torch.linspace(0, N, N, dtype=dtype, device=device)
        ref_path.div_(N - 1.0).mul_(2.0).sub_(1.0)
        ref = ref_path[None, ...].expand(B, -1).flatten(1)
        return ref

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table', 'rpe_table'}

    def forward(self, x):
        dtype, device = x.dtype, x.device
        B, C, H, W = x.size()
        N = H * W

        # ---------------------------------------------------------
        # Predict [dx, dy, dt, g]
        # ---------------------------------------------------------
        offset_all = self.conv_offset(x).contiguous()  # [B, 4, H, W]
        offset, de_index, gate = torch.split(offset_all, [2, 1, 1], dim=1)

        Hk, Wk = offset.size(2), offset.size(3)
        offset_range = torch.tensor(
            [1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)],
            device=device,
            dtype=offset.dtype
        ).reshape(1, 2, 1, 1)

        # ---------------------------------------------------------
        # Spatial sampling positions
        # ---------------------------------------------------------
        offset = torch.tanh(offset) * offset_range * gate
        offset = einops.rearrange(offset, 'b p h w -> b h w p').contiguous()
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)
        pos = offset + reference

        # ---------------------------------------------------------
        # Ordering / path positions
        # ---------------------------------------------------------
        de_index = (torch.tanh(de_index) * gate).flatten(1)
        path_reference = self._get_path_ref_points(N, B, dtype, device)
        path_pos = de_index + path_reference

        # ---------------------------------------------------------
        # DP-style spatial sampling
        # ---------------------------------------------------------
        x_sampled = F.grid_sample(
            input=x,
            grid=pos[..., (1, 0)],  # y, x -> x, y
            mode='bilinear',
            align_corners=True
        )  # [B, C, H, W]

        # ---------------------------------------------------------
        # Position Compensation (PC)
        # ---------------------------------------------------------
        if self.use_pc:
            rpe_table = self.rpe_table
            rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
            rpe_bias = F.interpolate(
                rpe_bias, size=(H, W),
                mode='bilinear',
                align_corners=False
            )
            key_grid = self._get_key_ref_points(H, W, B, dtype, device)
            displacement = (key_grid - pos) * 0.5
            pos_bias = F.grid_sample(
                input=rpe_bias,
                grid=displacement[..., (1, 0)],
                mode='bilinear',
                align_corners=True
            )
            x = x_sampled + pos_bias
        else:
            x = x_sampled

        # ---------------------------------------------------------
        # DT-style reordering
        # ---------------------------------------------------------
        x = x.flatten(2)
        x, indices = DeformablePathTrans.apply(x, path_pos)  # [B, N, C]

        if self.debug:
            print(
                f"[DeformableLayer] "
                f"use_ep={self.use_ep}, use_mc={self.use_mc}, "
                f"use_pc={self.use_pc}, use_sa={self.use_sa}"
            )

        return x, indices


class DeformableLayerReverse(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, indices=None):
        x = x.flatten(2)
        B, C, N = x.size()
        index_re = torch.zeros_like(indices, device=x.device)
        index_re.scatter_add_(1, indices,
                              torch.arange(indices.size(-1), device=x.device).unsqueeze(0).expand(indices.size(0), -1))
        x = torch.gather(x, 2, index_re.unsqueeze(1).expand(-1, C, -1))
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = partial(nn.Conv2d, kernel_size=1, padding=0) if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def x_selective_scan(
        x: torch.Tensor = None,
        x_proj_weight: torch.Tensor = None,
        x_proj_bias: torch.Tensor = None,
        dt_projs_weight: torch.Tensor = None,
        dt_projs_bias: torch.Tensor = None,
        A_logs: torch.Tensor = None,
        Ds: torch.Tensor = None,
        out_norm: torch.nn.Module = None,
        nrows=-1,
        delta_softplus=True,
        to_dtype=True,
        force_fp32=True,
        stage=0,
        DS=None,
        DR=None,
        **kwargs,
):
    # out_norm: whatever fits (B, L, C); LayerNorm; Sigmoid; Softmax(dim=1);...
    K, D, R = dt_projs_weight.shape
    B, D, H, W = x.shape
    L = H * W
    _, N = A_logs.shape

    xs = x.new_empty((B, 3, D, H * W))
    xs[:, 0] = x.flatten(2, 3)
    xs[:, 1] = torch.flip(xs[:, 0], dims=[-1])
    temp, indices = DS(x)
    xs[:, 2] = temp.permute(0, 2, 1)

    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)
    xs = xs.view(B, -1, L)
    dts = dts.contiguous().view(B, -1, L)
    As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
    Bs = Bs.contiguous()
    Cs = Cs.contiguous()
    Ds = Ds.to(torch.float)  # (K * c)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)

    if force_fp32:
        xs = xs.to(torch.float)
        dts = dts.to(torch.float)
        Bs = Bs.to(torch.float)
        Cs = Cs.to(torch.float)

    def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
        return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

    ys: torch.Tensor = selective_scan(
        xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
    )

    ys = ys.view(B, K, -1, H, W)
    ys = ys.view(B, K, D, -1)
    y = (ys[:, 0] + ys[:, 1].flip(dims=[-1]) + DR(ys[:, 2], indices)) / 3.

    y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
    if K != 1: y = y.view(B, H, W, -1)

    return (y.to(x.dtype) if to_dtype else y), {'ys': ys, 'xs': xs, 'dts': dts, 'As': A_logs, 'Bs': Bs, 'Cs': Cs,
                                                'Ds': Ds, 'delta_bias': delta_bias}


def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """
    import numpy as np

    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop

    assert not with_complex

    flops = 0  # below code flops = 0
    if False:
        ...
        """
        dtype_in = u.dtype
        u = u.float()
        delta = delta.float()
        if delta_bias is not None:
            delta = delta + delta_bias[..., None].float()
        if delta_softplus:
            delta = F.softplus(delta)
        batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
        is_variable_B = B.dim() >= 3
        is_variable_C = C.dim() >= 3
        if A.is_complex():
            if is_variable_B:
                B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
            if is_variable_C:
                C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
        else:
            B = B.float()
            C = C.float()
        x = A.new_zeros((batch, dim, dstate))
        ys = []
        """

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
    if False:
        ...
        """
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        if not is_variable_B:
            deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
        else:
            if B.dim() == 3:
                deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
            else:
                B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
                deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
        if is_variable_C and C.dim() == 4:
            C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
        last_state = None
        """

    in_for_flops = B * D * N
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops
    if False:
        ...
        """
        for i in range(u.shape[2]):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            if not is_variable_C:
                y = torch.einsum('bdn,dn->bd', x, C)
            else:
                if C.dim() == 3:
                    y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
                else:
                    y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
            if i == u.shape[2] - 1:
                last_state = x
            if y.is_complex():
                y = y.real * 2
            ys.append(y)
        y = torch.stack(ys, dim=2) # (batch dim L)
        """

    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    if False:
        ...
        """
        out = y if D is None else y + u * rearrange(D, "d -> d 1")
        if z is not None:
            out = out * F.silu(z)
        out = out.to(dtype=dtype_in)
        """
    return flops


def print_jit_input_names(inputs):
    print("input params: ", end=" ", flush=True)
    try:
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)




