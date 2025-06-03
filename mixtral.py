import math
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TokenEmbedding(nn.Module):
    """
    Simple token embedding + √(d_model) scaling.
    """
    def __init__(self, vocab_size: int, emb_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.scale = math.sqrt(emb_size)

    def forward(self, tokens: torch.LongTensor) -> torch.Tensor:
        # tokens: (batch, seq_len), dtype=torch.long
        # returns (batch, seq_len, emb_size)
        return self.embedding(tokens) * self.scale


class InputEmbedding(nn.Module):
    """
    Input embedding = TokenEmbedding + Dropout.
    (No separate positional embedding here because we use RoPE in attention.)
    """
    def __init__(self, vocab_size: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.token = TokenEmbedding(vocab_size, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        # x: (batch, seq_len) long
        tok_emb = self.token(x)        # (batch, seq_len, d_model)
        return self.dropout(tok_emb)   # (batch, seq_len, d_model)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    For RoPE: split the last dimension in half, then produce (−x₂, x₁).
    If x has shape (..., 2*k), then:
       x[..., :k] = x₁,  x[..., k:] = x₂
    return concat(−x₂, x₁) along last dim.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def build_rotary_pos_emb(
        dim: int,
        seq_len: int,
        device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build RoPE cosine and sine embeddings ahead of time.

    Args:
      dim: head_dim (must be even).
      seq_len: maximum sequence length.
      device: where to place the resulting buffers.

    Returns:
      cos: shape (1, 1, seq_len, dim)
      sin: shape (1, 1, seq_len, dim)
    """
    inv_freq = 1.0 / (
            10000 ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
    )  # (dim/2,)
    t = torch.arange(seq_len, device=device, dtype=torch.float32)  # (seq_len,)
    freqs = torch.einsum('i,j->ij', t, inv_freq)  # (seq_len, dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)        # (seq_len, dim)
    cos = emb.cos()[None, None, :, :]              # (1, 1, seq_len, dim)
    sin = emb.sin()[None, None, :, :]              # (1, 1, seq_len, dim)
    return cos, sin


class GqaAndSwa(nn.Module):
    """
    Grouped‐Query Attention + Sliding‐Window Attention + RoPE.

    - d_model: model dimension
    - num_heads: total number of attention heads
    - num_kv_heads: number of (grouped) KV heads
    - window_size: sliding window size (each token attends only to previous window_size tokens,
                   except when overridden by global_token_mask)
    - max_seq_len: maximum sequence length (for rotary pos embeddings)
    - dropout: dropout after softmax and after out_proj
    """
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

        # Projections for Q, K, V, and output
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Build rotary embeddings **directly on CPU** (model will move them to GPU if needed)
        cos_buf, sin_buf = build_rotary_pos_emb(self.head_dim, max_seq_len, device=torch.device('cpu'))
        self.register_buffer("rotary_cos", cos_buf, persistent=True)
        self.register_buffer("rotary_sin", sin_buf, persistent=True)

        # A small cache (dict) for sliding‐window masks: keys are seq_len (int), values are tensors.
        self._mask_cache = {}

    def _make_sliding_window_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Return a (seq_len×seq_len) mask where positions outside the
        sliding window (i - j not in [0, window_size)) are set to −inf, else 0.0.
        Cache masks by seq_len to avoid repeated creation.

        Result has shape (seq_len, seq_len).
        """
        if seq_len not in self._mask_cache:
            # Create it once on CPU, then move to the requested device when returning.
            i = torch.arange(seq_len, device='cpu').unsqueeze(1)  # (seq_len, 1)
            j = torch.arange(seq_len, device='cpu').unsqueeze(0)  # (1, seq_len)
            diff = i - j  # (seq_len, seq_len)
            base_mask = torch.where(
                (diff < 0) | (diff >= self.window_size),
                float("-inf"),
                0.0
            )  # (seq_len, seq_len)
            self._mask_cache[seq_len] = base_mask  # store on CPU
        # Move to the correct device/dtype each time
        return self._mask_cache[seq_len].to(device)

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            global_token_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        mask: optional causal mask or other mask of shape (batch, 1, seq_len, seq_len)
        global_token_mask: optional boolean mask (batch, seq_len),
                           if True at position i, that token can attend to ALL other tokens.
        Returns: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.size()
        device = x.device

        # 1) Project Q: (batch, seq_len, d_model) → (batch, seq_len, num_heads, head_dim)
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)  # (batch, heads, seq_len, head_dim)

        # 2) Project K & V with grouped KV heads
        #   - First linear to (batch, seq_len, d_model), then reshape
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.kv_head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.kv_head_dim)

        # We want to expand each KV‐head into `num_heads / num_kv_heads` heads of size head_dim:
        # Current k: (batch, seq_len, num_kv_heads, kv_head_dim)
        # Reshape pipeline: permute → split into groups → reshape
        group_size = self.num_heads // self.num_kv_heads

        k = k.permute(0, 2, 1, 3)  # (batch, kv_heads, seq_len, kv_head_dim)
        v = v.permute(0, 2, 1, 3)

        # Now break the kv_head_dim into (group_size, head_dim)
        # kv_head_dim == group_size * head_dim
        k = k.view(batch_size, self.num_kv_heads, seq_len, group_size, self.head_dim)
        v = v.view(batch_size, self.num_kv_heads, seq_len, group_size, self.head_dim)

        # Permute so that we get (batch, kv_heads, group_size, seq_len, head_dim)
        # → then collapse (kv_heads × group_size) = num_heads
        k = k.permute(0, 1, 3, 2, 4).contiguous().view(batch_size, self.num_heads, seq_len, self.head_dim)
        v = v.permute(0, 1, 3, 2, 4).contiguous().view(batch_size, self.num_heads, seq_len, self.head_dim)

        # 3) Apply RoPE to q and k
        #    - rotary_cos/sin have shape (1, 1, max_seq_len, head_dim)
        cos = self.rotary_cos[:, :, :seq_len, :].to(device)  # (1,1,seq_len,head_dim)
        sin = self.rotary_sin[:, :, :seq_len, :].to(device)
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)

        # 4) Compute attention scores: (batch, heads, seq_len, head_dim) × (batch, heads, head_dim, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (batch, heads, seq_len, seq_len)

        # 5) Build sliding‐window mask (seq_len×seq_len) once, then broadcast
        sliding_mask = self._make_sliding_window_mask(seq_len, device)
        sliding_mask = sliding_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

        # 6) Combine with any provided mask
        if mask is not None:
            combined_mask = mask + sliding_mask  # shape (batch, 1, seq_len, seq_len)
        else:
            combined_mask = sliding_mask         # shape (1, 1, seq_len, seq_len)

        # 7) Override sliding window for any “global” tokens
        if global_token_mask is not None:
            # global_token_mask: (batch, seq_len) bool
            gt = global_token_mask.unsqueeze(1).unsqueeze(3)  # (batch, 1, seq_len, 1)
            gs = global_token_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
            combined_mask = torch.where(
                (gt | gs),  # if either row‐token or column‐token is “global” → 0.0
                torch.zeros_like(combined_mask),  # allow full attention
                combined_mask
            )

        scores = scores + combined_mask  # apply mask → (batch, heads, seq_len, seq_len)

        # 8) Softmax + dropout
        attn = F.softmax(scores, dim=-1)  # (batch, heads, seq_len, seq_len)
        attn = self.dropout(attn)

        # 9) Weighted sum: attn @ v → (batch, heads, seq_len, head_dim)
        out = torch.matmul(attn, v)  # (batch, heads, seq_len, head_dim)

        # 10) Rearrange → (batch, seq_len, d_model)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.d_model)

        # 11) Final linear + dropout
        out = self.out_proj(out)
        out = self.dropout(out)
        return out


class RMSNorm(nn.Module):
    """
    Root‐Mean‐Square LayerNorm (no centering, just scaling).
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()  # (..., 1)
        return (x / rms) * self.weight  # scale by learned weight (broadcasted)


class SublayerConnection(nn.Module):
    """
    Pre‐Norm + Residual + Dropout
    """
    def __init__(self, size: int, dropout: float):
        super().__init__()
        self.norm = RMSNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer: callable) -> torch.Tensor:
        # Apply RMSNorm, pass through sublayer, apply dropout, then residual‐add
        return x + self.dropout(sublayer(self.norm(x)))


class SwiGLU(nn.Module):
    """
    SiLU‐based Gated Linear Unit:
      input x of shape (..., 2*d_ff) → chunk into (x1: ..., d_ff) and (x2: ..., d_ff)
      output = x1 * SiLU(x2)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return x1 * F.silu(x2)


class PositionwiseFeedForward(nn.Module):
    """
    Standard Mistral‐style FFN (two‐layer feed‐forward with SwiGLU activation).
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # Project (d_model → 2*d_ff) → SwiGLU → (d_ff → d_model)
        self.linear1 = nn.Linear(d_model, d_ff * 2, bias=False)
        self.activation = SwiGLU()
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x can be (batch, seq_len, d_model) or (tokens, d_model)
        x = self.linear1(x)       # → (..., 2*d_ff)
        x = self.activation(x)    # → (..., d_ff)
        x = self.linear2(x)       # → (..., d_model)
        return self.dropout(x)    # → (..., d_model)


class MoEPositionwiseFeedForward(nn.Module):
    """
    Sparse Mixture‐of‐Experts Feedforward (Mixtral style).
    - Noisy Top‐K Gating
    - Capacity control (limit tokens per expert)
    - Returns (combined_output, aux_loss)
    """
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

        # 1) Create `num_experts` separate SwiGLU feedforward modules
        self.experts = nn.ModuleList([
            PositionwiseFeedForward(d_model, d_ff, dropout)
            for _ in range(num_experts)
        ])

        # 2) Gating network: from d_model → num_experts logits
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x: (batch, seq_len, d_model)
        Returns:
          combined: (batch, seq_len, d_model)
          aux_loss: scalar tensor for load balancing
        """
        batch_size, seq_len, d_model = x.shape
        assert d_model == self.d_model

        total_tokens = batch_size * seq_len
        flat_x = x.view(total_tokens, d_model)  # (T, d_model), T = batch × seq_len

        # 1) Compute gating logits
        gate_logits = self.gate(flat_x)  # (T, E)
        if self.training and self.noise_std > 0.0:
            noise = torch.randn_like(gate_logits) * self.noise_std
            gate_logits = gate_logits + noise

        gate_scores = F.softmax(gate_logits, dim=-1)  # (T, E)
        topk_vals, topk_idx = gate_scores.topk(self.top_k, dim=-1)  # (T, K)

        # 2) Auxiliary load‐balancing loss: encourage equal expert usage
        phi_mean = gate_scores.mean(dim=0)  # (E,)
        aux_loss = self.num_experts * torch.sum(phi_mean * phi_mean)

        # 3) Determine capacity = max(1, floor(capacity_factor * (T / num_experts)))
        capacity = max(1, int(self.capacity_factor * total_tokens / self.num_experts))

        # 4) Create a zero‐tensor for combined outputs, same dtype/device as flat_x
        combined_flat = torch.zeros_like(flat_x)  # (T, d_model)

        # 5) For each expert e, collect tokens routed to e (up to capacity), apply FFN, accumulate
        #    We compute a “weight” per token = sum of its topk gating probabilities for that expert.
        for e in range(self.num_experts):
            # Find all token‐positions where e is in the top‐k indices
            mask_e = (topk_idx == e).any(dim=1)  # (T,) boolean
            if not mask_e.any():
                continue

            positions_e = mask_e.nonzero(as_tuple=False).squeeze(-1)  # (n_e,)
            n_e = positions_e.size(0)

            # Compute the combined gate‐weight for each position among those in positions_e
            #   For each token i, its weight for expert e = sum over k∈[0,K) of topk_vals[i,k] if topk_idx[i,k]==e
            one_hot = (topk_idx == e).float()               # (T, K)
            weights_per_token = (one_hot * topk_vals).sum(dim=1)  # (T,)
            weights_e_full = weights_per_token[positions_e]       # (n_e,)

            # Enforce capacity: if n_e > capacity, pick the top‐capacity tokens by weight
            if n_e > capacity:
                top_indices = torch.argsort(weights_e_full, descending=True)[:capacity]
                positions_e = positions_e[top_indices]
                weights_e_full = weights_e_full[top_indices]
                n_e = capacity

            if n_e == 0:
                continue

            # Now gather inputs and weights for expert e
            inputs_e = flat_x[positions_e]         # (n_e, d_model)
            weights_e = weights_e_full.unsqueeze(-1)  # (n_e, 1)
            outputs_e = self.experts[e](inputs_e)    # (n_e, d_model)
            # Weighted accumulation
            combined_flat[positions_e] += outputs_e * weights_e  # (n_e, d_model)

        combined = combined_flat.view(batch_size, seq_len, d_model)  # (batch, seq_len, d_model)
        return combined, aux_loss


class DecoderBlockMoE(nn.Module):
    """
    One decoder block with:
     - GQA + Sliding‐Window Attention (pre‐norm)
     - Sparse MoE feed‐forward (pre‐norm)
    Returns (output_tensor, aux_loss) where aux_loss is from the MoE layer.
    """
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
        # Two pre‐norm + dropout + residual sublayers:
        self.sublayers = nn.ModuleList([
            SublayerConnection(dim, dropout),  # for self‐attention
            SublayerConnection(dim, dropout),  # for MoE feed‐forward
        ])

    def forward(
            self,
            x: torch.Tensor,
            causal_mask: Optional[torch.Tensor] = None,
            global_token_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch, seq_len, dim)
        causal_mask: (batch, 1, seq_len, seq_len) or None
        global_token_mask: (batch, seq_len) bool or None
        Returns: (out, aux_loss)
         - out: (batch, seq_len, dim)
         - aux_loss: scalar tensor
        """
        # 1) Self‐attention block
        attn_out = self.sublayers[0](x, lambda _x: self.attn(_x, mask=causal_mask, global_token_mask=global_token_mask))
        # 2) MoE feed‐forward block
        x_norm = self.sublayers[1].norm(attn_out)
        moe_out, aux_loss = self.ff(x_norm)  # (batch, seq_len, dim), scalar
        out = attn_out + self.sublayers[1].dropout(moe_out)

        return out, aux_loss


class Mixtral7B(nn.Module):
    """
    Mixtral 7B: Mistral 7B + Sparse MoE (Top‐K Noisy Gating + Capacity Control)
    """
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

        # 1) Input embedding
        self.embed = InputEmbedding(vocab_size, dim, dropout)

        # 2) Stack of N decoder blocks
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

        # 3) Final RMSNorm + LM head
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        # Weight‐tying: LM head’s weight = token embedding’s weight
        self.head.weight = self.embed.token.embedding.weight

    def forward(
            self,
            x: torch.LongTensor,
            global_token_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x: (batch, seq_len) input token IDs
          global_token_mask: optional (batch, seq_len) bool
        Returns:
          logits: (batch, seq_len, vocab_size)
          total_aux_loss: scalar tensor (sum of MoE auxiliary losses)
        """
        batch_size, seq_len = x.shape
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max length {self.max_len}")

        # 1) Embed tokens → (batch, seq_len, dim)
        x_emb = self.embed(x)

        # 2) Build causal mask: strictly causal (i cannot attend to j > i)
        #    **수정**: x.embeddings.device 대신 x.device 사용
        device = x.device
        causal = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1
        )  # (seq_len, seq_len)
        causal_mask = causal.unsqueeze(0).unsqueeze(0)           # (1, 1, seq_len, seq_len)
        causal_mask = causal_mask.expand(batch_size, -1, -1, -1)  # (batch, 1, seq_len, seq_len)

        total_aux_loss = torch.tensor(0.0, device=x.device)

        # 3) Pass through each decoder block, accumulate aux losses
        hidden = x_emb
        for layer in self.layers:
            hidden, aux_loss = layer(hidden, causal_mask=causal_mask, global_token_mask=global_token_mask)
            total_aux_loss = total_aux_loss + aux_loss

        # 4) Final RMSNorm
        hidden = self.norm(hidden)  # (batch, seq_len, dim)

        # 5) LM head → logits
        logits = self.head(hidden)  # (batch, seq_len, vocab_size)
        return logits, total_aux_loss
