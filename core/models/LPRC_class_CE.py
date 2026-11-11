import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# Part 1. LPRC: Local Pairwise Rank Consistency
# =========================================================

def _pad_last_token_2d(x, pad_n):
    # x: [B, N]
    if pad_n == 0:
        return x
    pad = x[:, -1:].expand(-1, pad_n)
    return torch.cat([x, pad], dim=1)


def split_windows_1d_scores(s, window_size):
    # s: [B, N]
    B, N = s.shape
    pad_n = (window_size - (N % window_size)) % window_size
    s = _pad_last_token_2d(s, pad_n)
    N_pad = s.shape[1]
    num_win = N_pad // window_size
    sw = s.view(B, num_win, window_size)
    return sw, pad_n


class LPRCGradCompensator(nn.Module):
    """
    Local Pairwise Rank Consistency (LPRC)

    作用：
    1) 不改 hard-sort forward
    2) 只在训练时，对 path_pos / de_index 补一个局部 pairwise ranking 梯度
    3) 推理阶段 0 额外开销

    用法：
        path_pos = self.lprc_comp.attach(path_pos)
    """

    def __init__(
        self,
        window_tokens=16,
        pairs_per_window=24,
        tau=1.0,
        margin=0.0,
        weight=0.05,
    ):
        super().__init__()
        self.window_tokens = int(window_tokens)
        self.pairs_per_window = int(pairs_per_window)
        self.tau = float(tau)
        self.margin = float(margin)
        self.weight = float(weight)

        self.last_aux_loss = None
        self.last_grad_norm = None

    def _build_pair_index(self, M: int, device):
        i_idx, j_idx = torch.triu_indices(M, M, offset=1, device=device)
        if i_idx.numel() == 0:
            return None, None

        if self.pairs_per_window > 0 and i_idx.numel() > self.pairs_per_window:
            step = max(1, i_idx.numel() // self.pairs_per_window)
            keep = torch.arange(0, i_idx.numel(), step, device=device)[:self.pairs_per_window]
            i_idx = i_idx[keep]
            j_idx = j_idx[keep]

        return i_idx, j_idx

    def pairwise_rank_loss(self, score: torch.Tensor):
        """
        score: [B, N]
        用当前 hard order(detach 后) 作为局部成对顺序标签，
        再对 score difference 加一个连续可导的 ranking loss。
        """
        if score.dim() != 2:
            score = score.view(score.shape[0], -1)

        B, N = score.shape
        if N < 2:
            return None

        ws = min(self.window_tokens, N)
        if ws < 2:
            return None

        sw, _ = split_windows_1d_scores(score, ws)  # [B, num_win, ws]
        if sw.numel() == 0:
            return None

        i_idx, j_idx = self._build_pair_index(ws, score.device)
        if i_idx is None:
            return None

        # 当前 hard 排序结果（停止梯度）
        order = torch.argsort(sw.detach(), dim=-1, descending=False)  # [B, num_win, ws]
        rank = torch.argsort(order, dim=-1)                           # [B, num_win, ws]

        # y = +1 表示 i 在 j 前面；否则 y = -1
        y = torch.where(
            rank[..., i_idx] < rank[..., j_idx],
            torch.ones(1, device=score.device, dtype=score.dtype),
            -torch.ones(1, device=score.device, dtype=score.dtype)
        )

        diff = sw[..., i_idx] - sw[..., j_idx]

        if self.margin > 0.0:
            # hinge ranking
            loss = F.relu(self.margin - y * diff)
        else:
            # logistic ranking
            tau = max(self.tau, 1e-6)
            loss = F.softplus(-(y * diff) / tau)

        return loss.mean()

    def attach(self, score: torch.Tensor):
        """
        给 score 注册 hook:
            grad <- grad + weight * grad(LPRC, score)

        forward 数值不变，只在 backward 时补梯度
        """
        self.last_aux_loss = None
        self.last_grad_norm = None

        if (not self.training) or self.weight <= 0.0:
            return score

        if (not isinstance(score, torch.Tensor)) or (not score.requires_grad):
            return score

        aux_loss = self.pairwise_rank_loss(score)
        if aux_loss is None:
            return score

        grad_aux = torch.autograd.grad(
            aux_loss,
            score,
            retain_graph=True,
            create_graph=False,
            allow_unused=True
        )[0]

        if grad_aux is None:
            return score

        grad_aux = grad_aux.detach()
        self.last_aux_loss = aux_loss.detach()
        self.last_grad_norm = grad_aux.norm().detach()

        score.register_hook(
            lambda grad, ga=grad_aux, w=self.weight: grad + w * ga
        )

        return score
