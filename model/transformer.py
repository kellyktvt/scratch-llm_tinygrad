from typing import Tuple
from tinygrad import Tensor, dtypes
from tinygrad.nn import RMSNorm, Linear

# Specify machine epsilon for 'Tensor.float32'
EPS = Tensor([1.1920929e-07], dtype=dtypes.float32).item()


# Computes rotary positional encodings for each position in a sequence
class RotaryPositionalEncoding():
    # base = base val for positional encoding; eps = small epsilon val to avoid division by 0 in scaling
    def __init__(self, seq_len: int, dim_emb: int, base: int = 10000, eps: float = EPS) -> None:
        self.dim_emb = dim_emb
        # Generates 'indices' as a Tensor tensor repping positions in the seq
        indices = Tensor.arange(0, seq_len, dtype=dtypes.float32)
        # Computes'scale' vals for scaling rotations based on 'base' and 'dim_emb'
        scale = 1 / (base ** (Tensor.arange(0, dim_emb, 2, dtype=dtypes.float32) / dim_emb) + eps)

        # Construct 'position' tensor by outer product of 'indices' and 'scale'
        position = indices.reshape(-1, 1) * scale.reshape(1, -1) 
        # , concatenated along the last dimension
        position = position.cat(position, dim=-1)

        # Compute 'position_cos' and 'position_sin' tensors using cosine and sine fxns applied to 'position'
        self.position_cos = Tensor.cos(position[None, None, :, :])
        self.position_sin = Tensor.sin(position[None, None, :, :])

        self.position_cos.requires_grad = False
        self.position_sin.requires_grad = False

    # Method to perform a specific rotation operation on a tensor 'x'
    def _rotate_half(self, x: Tensor) -> Tensor:
        # Split 'x' into 2 halves along the last dimension based on 'dim_emb'
        x1, x2 = x[..., : self.dim_emb // 2], x[..., self.dim_emb // 2 :]

        # Concatenate '(-x2, x1)' along the last dimension and return result
        return x1.cat(-x2, dim=-1)

    # Method to apply rotary positional encodings to the input tensor 'x'
    def __call__(self, x: Tensor) -> Tensor:
        self.position_cos.requires_grad = False
        self.position_sin.requires_grad = False
        
        # x is of shape  (bs, num_heads, seq_len, dim_emb)
        x = (x * self.position_cos) + (self._rotate_half(x) * self.position_sin)

        self.position_cos.requires_grad = False
        self.position_sin.requires_grad = False

        return x


# Implement SwiGLU activation function using single linear layer
class SwiGLU():
    # SwiGLU(x) = (xW + b) ⊗ swish(xZ + c) where W, Z, b, c are learnable params
    # Initialize the linear transformation
    def __init__(self, dim_in: int, bias: bool = True) -> None:
        self.dim_in = dim_in
        self.linear = Linear(dim_in, 2 * dim_in, bias=bias)

    # Method to apply SwiGLU activation to input tensor 'x'
    def __call__(self, x: Tensor) -> Tensor:
        # uses only one weight matrix instead of two
        x = self.linear(x)
        x = x[..., : self.dim_in].silu() + x[..., self.dim_in :]

        return x


class MultiHeadAttention():
    def __init__(self, seq_len: int, num_heads: int, dim_emb: int, dim_k: int = None, dim_v: int = None, causal=True) -> None:
        assert dim_emb % num_heads == 0, "num_heads must be a multiple of dim_emb"

        self.seq_len = seq_len
        self.num_heads = num_heads
        self.dim_head = dim_emb // num_heads
        self.dim_k = dim_k or dim_emb
        self.dim_v = dim_v or dim_emb
        self.causal = causal

        # positional encoding to be applied to query and key projections
        self.positional_encoding = RotaryPositionalEncoding(seq_len, dim_emb // num_heads)

        # Query, Key and Value projections batched into one linear layer
        self.proj_qkv = Linear(dim_emb, 3 * dim_emb, bias=False)
        self.proj_out = Linear(self.dim_v, self.dim_v, bias=False)

        # Build the causal mask, masking upper triangular part of attention scores
        self.causal_mask = Tensor.triu(Tensor.ones(seq_len, seq_len), diagonal=1).bool()
        # Ensure causal_mask does not require gradients 
        self.causal_mask.requires_grad = False

    # Method to perform forward pass of MultiHeadAttention layer
    def __call__(self, x: Tensor, return_scores: bool = False) -> Tensor | Tuple[Tensor, Tensor]:
        self.causal_mask.requires_grad = False

        # projects input to Q, K, V spaces
        qkv = self.proj_qkv(x)  # (bs, seq_len, 3 * dim_emb)

        # split into Q, K, V
        q, k, v = qkv.chunk(3, dim=-1)  # (bs, seq_len, dim_k), (bs, seq_len, dim_k), (bs, seq_len, dim_v)

        # split projections between heads -> (bs, num_heads, seq_len, dim_k)
        q = q.reshape(q.shape[0], q.shape[1], self.num_heads, self.dim_head).permute(0, 2, 1, 3)
        k = k.reshape(k.shape[0], k.shape[1], self.num_heads, self.dim_head).permute(0, 2, 1, 3)
        v = v.reshape(v.shape[0], v.shape[1], self.num_heads, self.dim_head).permute(0, 2, 1, 3)

        # apply positional encoding to projections, for each heads
        q = self.positional_encoding(q)  # (bs, num_heads, seq_len, dim_k)
        k = self.positional_encoding(k)  # (bs, num_heads, seq_len, dim_k)

        # Compute the correlation between a query q_i and all the keys, for every q_i
        attn_scores = (q @ k.permute(0, 1, 3, 2)) * self.dim_k**-0.5  # (bs, num_heads, seq_len, seq_len)

        # Fill the upper triangular part of the attention scores with -inf to inhibit them in the softmax
        if self.causal:
            attn_scores = attn_scores.masked_fill(self.causal_mask[None, None, ...], float('-inf'))
            attn_scores.requires_grad = False

        # attention scores are used to build a weighted linear combination of values vectors
        attn_scores = Tensor.softmax(attn_scores, axis=-1)  # (bs, num_heads, seq_len, seq_len)
        out = attn_scores @ v  # (bs, num_heads, seq_len, dim_v)

        # Merge heads by reshaping and permuting output tensor
        out = out.permute(0, 2, 1, 3).reshape(-1, self.seq_len, self.dim_v)  # (bs, seq_len, dim_v)

        # projects to the output space
        out = self.proj_out(out)  # (bs, seq_len, dim_v)

        # Ensure causal_mask does not require gradients at the end of the forward pass
        self.causal_mask.requires_grad = False

        # Return output tensor and optionally the attention scores if 'return_scores' is True
        if return_scores:
            return out, attn_scores
        else:
            return out


# Defines feedforward layer sequentially
class FeedForward():
    def __init__(self, dim_in: int, dim_hidden: int, bias: bool = False) -> None:
        self.linear1 = Linear(dim_in, dim_hidden, bias=bias)   # First Linear layer
        self.swiglu = SwiGLU(dim_hidden)                       # SwiGLU activation
        self.linear2 = Linear(dim_hidden, dim_in, bias=bias)   # Second Linear layer

    def __call__(self, x: Tensor) -> Tensor:
        return self.linear2(self.swiglu(self.linear1(x)))


class TransformerBlock():
    def __init__(self, seq_len: int, dim_emb: int, attn_num_heads: int, ffn_hidden_dim: int, ffn_bias: bool = False, attn_causal: bool = True, ) -> None:
        self.norm_attn = RMSNorm(dim_emb)
        self.multihead_attn = MultiHeadAttention(seq_len, attn_num_heads, dim_emb)
        self.norm_ffn = RMSNorm(dim_emb)
        self.feed_forward = FeedForward(dim_emb, ffn_hidden_dim, bias=ffn_bias)

    def __call__(self, x: Tensor) -> Tensor:
        x = x + self.multihead_attn(self.norm_attn(x))  # (bs, seq_len, dim_in)
        x = x + self.feed_forward(self.norm_ffn(x))  # (bs, seq_len, dim_in)
        return x  # (bs, seq_len, dim_in)
