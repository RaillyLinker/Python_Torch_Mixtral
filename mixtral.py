import math
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.scale = math.sqrt(emb_size)

    def forward(self, tokens: torch.LongTensor) -> torch.Tensor:
        return self.embedding(tokens) * self.scale


class InputEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.token = TokenEmbedding(vocab_size, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        tok_emb = self.token(x)
        return self.dropout(tok_emb)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def build_rotary_pos_emb(
        dim: int,
        seq_len: int,
        device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
            10000 ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
    )
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.einsum('i,j->ij', t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()[None, None, :, :]
    sin = emb.sin()[None, None, :, :]
    return cos, sin


class GqaAndSwa(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            num_kv_heads: int,
            window_size: int,
            max_seq_len: int,
            dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert d_model % num_kv_heads == 0, "d_model must be divisible by num_kv_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.kv_head_dim = d_model // num_kv_heads
        self.window_size = window_size
        self.dropout = nn.Dropout(dropout)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        cos_buf, sin_buf = build_rotary_pos_emb(self.head_dim, max_seq_len, device=torch.device('cpu'))
        self.register_buffer("rotary_cos", cos_buf, persistent=True)
        self.register_buffer("rotary_sin", sin_buf, persistent=True)

        self._mask_cache = {}

    def _make_sliding_window_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if seq_len not in self._mask_cache:
            i = torch.arange(seq_len, device='cpu').unsqueeze(1)
            j = torch.arange(seq_len, device='cpu').unsqueeze(0)
            diff = i - j
            base_mask = torch.where(
                (diff < 0) | (diff >= self.window_size),
                float("-inf"),
                0.0
            )
            self._mask_cache[seq_len] = base_mask
        return self._mask_cache[seq_len].to(device)

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            global_token_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        device = x.device

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)

        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.kv_head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.kv_head_dim)

        group_size = self.num_heads // self.num_kv_heads

        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        k = k.view(batch_size, self.num_kv_heads, seq_len, group_size, self.head_dim)
        v = v.view(batch_size, self.num_kv_heads, seq_len, group_size, self.head_dim)

        k = k.permute(0, 1, 3, 2, 4).contiguous().view(batch_size, self.num_heads, seq_len, self.head_dim)
        v = v.permute(0, 1, 3, 2, 4).contiguous().view(batch_size, self.num_heads, seq_len, self.head_dim)

        cos = self.rotary_cos[:, :, :seq_len, :].to(device)
        sin = self.rotary_sin[:, :, :seq_len, :].to(device)
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        sliding_mask = self._make_sliding_window_mask(seq_len, device)
        sliding_mask = sliding_mask.unsqueeze(0).unsqueeze(0)

        if mask is not None:
            combined_mask = mask + sliding_mask
        else:
            combined_mask = sliding_mask

        if global_token_mask is not None:
            gt = global_token_mask.unsqueeze(1).unsqueeze(3)
            gs = global_token_mask.unsqueeze(1).unsqueeze(2)
            combined_mask = torch.where(
                (gt | gs),
                torch.zeros_like(combined_mask),
                combined_mask
            )

        scores = scores + combined_mask

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)

        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.d_model)

        out = self.out_proj(out)
        out = self.dropout(out)
        return out


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


class SublayerConnection(nn.Module):
    def __init__(self, size: int, dropout: float):
        super().__init__()
        self.norm = RMSNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer: callable) -> torch.Tensor:
        return x + self.dropout(sublayer(self.norm(x)))


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return x1 * F.silu(x2)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff * 2, bias=False)
        self.activation = SwiGLU()
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return self.dropout(x)


class MoEPositionwiseFeedForward(nn.Module):
    def __init__(
            self,
            d_model: int,
            d_ff: int,
            num_experts: int,
            top_k: int = 2,
            dropout: float = 0.1,
            noise_std: float = 1.0,
            capacity_factor: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std
        self.capacity_factor = capacity_factor

        self.experts = nn.ModuleList([
            PositionwiseFeedForward(d_model, d_ff, dropout)
            for _ in range(num_experts)
        ])

        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, d_model = x.shape
        assert d_model == self.d_model

        total_tokens = batch_size * seq_len
        flat_x = x.view(total_tokens, d_model)

        gate_logits = self.gate(flat_x)
        if self.training and self.noise_std > 0.0:
            noise = torch.randn_like(gate_logits) * self.noise_std
            gate_logits = gate_logits + noise

        gate_scores = F.softmax(gate_logits, dim=-1)
        topk_vals, topk_idx = gate_scores.topk(self.top_k, dim=-1)

        phi_mean = gate_scores.mean(dim=0)
        aux_loss = self.num_experts * torch.sum(phi_mean * phi_mean)

        capacity = max(1, int(self.capacity_factor * total_tokens / self.num_experts))

        combined_flat = torch.zeros_like(flat_x)

        for e in range(self.num_experts):
            mask_e = (topk_idx == e).any(dim=1)
            if not mask_e.any():
                continue

            positions_e = mask_e.nonzero(as_tuple=False).squeeze(-1)
            n_e = positions_e.size(0)

            one_hot = (topk_idx == e).float()
            weights_per_token = (one_hot * topk_vals).sum(dim=1)
            weights_e_full = weights_per_token[positions_e]

            if n_e > capacity:
                top_indices = torch.argsort(weights_e_full, descending=True)[:capacity]
                positions_e = positions_e[top_indices]
                weights_e_full = weights_e_full[top_indices]
                n_e = capacity

            if n_e == 0:
                continue

            inputs_e = flat_x[positions_e]
            weights_e = weights_e_full.unsqueeze(-1)
            outputs_e = self.experts[e](inputs_e)
            combined_flat[positions_e] += outputs_e * weights_e

        combined = combined_flat.view(batch_size, seq_len, d_model)
        return combined, aux_loss


class DecoderBlockMoE(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            num_kv_heads: int,
            hidden_dim: int,
            num_experts: int,
            top_k: int = 2,
            window_size: int = 4096,
            dropout: float = 0.1,
            noise_std: float = 1.0,
            capacity_factor: float = 1.0,
            max_seq_len: int = 131072,
    ):
        super().__init__()
        self.attn = GqaAndSwa(
            d_model=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            window_size=window_size,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )
        self.ff = MoEPositionwiseFeedForward(
            d_model=dim,
            d_ff=hidden_dim,
            num_experts=num_experts,
            top_k=top_k,
            dropout=dropout,
            noise_std=noise_std,
            capacity_factor=capacity_factor,
        )
        self.sublayers = nn.ModuleList([
            SublayerConnection(dim, dropout),
            SublayerConnection(dim, dropout),
        ])

    def forward(
            self,
            x: torch.Tensor,
            causal_mask: Optional[torch.Tensor] = None,
            global_token_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out = self.sublayers[0](x, lambda _x: self.attn(_x, mask=causal_mask, global_token_mask=global_token_mask))
        x_norm = self.sublayers[1].norm(attn_out)
        moe_out, aux_loss = self.ff(x_norm)
        out = attn_out + self.sublayers[1].dropout(moe_out)

        return out, aux_loss


class Mixtral7B(nn.Module):
    def __init__(
            self,
            vocab_size: int = 32000,
            dim: int = 4096,
            n_layers: int = 32,
            num_heads: int = 32,
            num_kv_heads: int = 8,
            hidden_dim: int = 14336,
            num_experts: int = 16,
            top_k: int = 2,
            max_len: int = 131072,
            window_size: int = 4096,
            dropout: float = 0.1,
            noise_std: float = 1.0,
            capacity_factor: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.max_len = max_len

        self.embed = InputEmbedding(vocab_size, dim, dropout)

        self.layers = nn.ModuleList([
            DecoderBlockMoE(
                dim=dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                hidden_dim=hidden_dim,
                num_experts=num_experts,
                top_k=top_k,
                window_size=window_size,
                dropout=dropout,
                noise_std=noise_std,
                capacity_factor=capacity_factor,
                max_seq_len=max_len,
            )
            for _ in range(n_layers)
        ])

        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.embed.token.embedding.weight

    def forward(
            self,
            x: torch.LongTensor,
            global_token_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = x.shape
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max length {self.max_len}")

        x_emb = self.embed(x)

        device = x.device
        causal = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1
        )
        causal_mask = causal.unsqueeze(0).unsqueeze(0)
        causal_mask = causal_mask.expand(batch_size, -1, -1, -1)

        total_aux_loss = torch.tensor(0.0, device=x.device)

        hidden = x_emb
        for layer in self.layers:
            hidden, aux_loss = layer(hidden, causal_mask=causal_mask, global_token_mask=global_token_mask)
            total_aux_loss = total_aux_loss + aux_loss

        hidden = self.norm(hidden)

        logits = self.head(hidden)
        return logits, total_aux_loss
