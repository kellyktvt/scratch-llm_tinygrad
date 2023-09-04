import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, Sequential, Embedding, Linear, Dropout

from model.transformer import TransformerBlock


# TODO: implement the auto-regressive generation
class LLM(Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        dim_emb: int,
        num_layers: int,
        attn_num_heads: int,
        attn_causal: bool = True,
        emb_dropout: float = 0.0,
        ffd_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.seq_len = seq_len
        self.token_embedding = Embedding(vocab_size, dim_emb)
        self.emb_dropout = Dropout(emb_dropout)
        self.transformer = Sequential(
            *[
                TransformerBlock(seq_len, dim_emb, attn_num_heads, attn_causal, ffd_dropout=ffd_dropout)
                for _ in range(num_layers)
            ]
        )
        self.projection_head = Linear(dim_emb, vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.token_embedding(x)  # (bs, seq_len, dim_emb)
        x = self.emb_dropout(x)  # (bs, seq_len, dim_emb)
        x = self.transformer(x)  # (bs, seq_len, dim_emb)
        x = self.projection_head(x)  # (bs, seq_len, vocab_size)

        return x  # (bs, seq_len, vocab_size)

    @torch.inference_mode()
    def generate(self, inputs, max_seq_len, temperature=1.0):
        for _ in range(max_seq_len):
            # make sure the sequence we're generating doesn't exceed model's sequence length
            inputs_cond = inputs if inputs.size(1) <= self.seq_len else inputs[:, -self.seq_len :]

            # get logits from model (only the last token in the sequence) and rescale them to get a probability distribution over the vocabulary
            logits = self(inputs_cond)[:, -1, :]  # (bs, 1, vocab_size)
            probs = F.softmax(logits / temperature, dim=-1)

            # sample the next token
            next_token = torch.multinomial(probs, num_samples=1)

            # append to the sequence being generated
            inputs = torch.cat((inputs, next_token), dim=-1)

        return inputs
