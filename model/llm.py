from tinygrad import Tensor
from tinygrad.nn import Embedding, Linear, RMSNorm
from model.transformer import TransformerBlock

# Function that takes probabilities and a threshold for top-p sampling
def sample_top_p(probs: Tensor, threshold: float) -> Tensor:
    # Sort probabilities in descending order along the last dimension & also return corresponding indices
    sorted_probs, sorted_indices = Tensor.sort(probs, descending=True)  # (bs, vocab_size), (bs, vocab_size)
    # Compute cumulative sums along the last dimension
    cumulative_probs = Tensor.cumsum(sorted_probs, dim=-1)  # (bs, vocab_size)

    # Create mask to filter out probabilities
    mask = cumulative_probs > threshold
    # virtually discard tokens with lower probability
    sorted_probs[mask] = 0.0  
    # Normalize probabilities to sum to 1.0
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)  # rescale to sum to 1.0

    # Sample indices based on normalized probabilities
    next_token = Tensor.multinomial(sorted_probs, num_samples=1)
    # Retrieve token indices corresponding to sampled probabilities
    next_token = Tensor.gather(sorted_indices, dim=-1, index=next_token)

    # Returns index of next token to be generated
    return next_token

class LLM:
    # Initialize the model with params
    def __init__(
        self,
        vocab_size: int, # Size of vocabulary
        seq_len: int, # Length of input sequence
        dim_emb: int, # Dimensionality of token embeddings
        num_layers: int, # Number of transformer layers
        attn_num_heads: int, # Number of attention heads in the transformer
        ffn_hidden_dim: int, # Dimensionality of the feedforward network hidden layers
        ffn_bias: bool = False, # Whether to use bias in feedforward layers
        emb_dropout: float = 0.0, # Dropout probability for token embeddings
    ) -> None: # Indicates method doesn't return anything

        # Store input sequence length
        self.seq_len = seq_len
        # Initialize embedding layer mapping tokens to 'dim_emb' dimensions
        self.token_embedding = Embedding(vocab_size, dim_emb)
        # Initialize dropout layer appled to token embeddings w/ dropout probability
        self.emb_dropout = emb_dropout
        # Create a list of TransformerBlocks instead of a Sequential object
        self.transformer_blocks = [
            TransformerBlock(seq_len, dim_emb, attn_num_heads, ffn_hidden_dim, ffn_bias)
            for _ in range(num_layers)
        ]
        # Initialize RMS normalization for normalizing transformer output w/ 'dim_emb' dimensions
        self.norm = RMSNorm(dim_emb)
        # Initialize linear layer projecting 'dim_emb' dimensions to 'vocab_size' dimensions
        self.projection_head = Linear(dim_emb, vocab_size)

        # Weight tying links embedding layer and output projection together to share parameters
        # Which can enhance training efficiency and model performance
        # https://paperswithcode.com/method/weight-tying
        #self.token_embedding.weight = self.projection_head.weight

    # Method defining the data flow through the model during the forward pass
    def __call__(self, x: Tensor) -> Tensor:
        # Embeds input tokens (x) using initialized embedding layer (self.token_embedding)
        x = self.token_embedding(x)  # resulting shape: (batch_size, seq_len, dim_emb)
        # Applies dropout to embedded tokens (x) to prevent overfitting during training
        x = x.dropout(self.emb_dropout)
        # Use the sequential method to apply transformer blocks
        x = x.sequential(self.transformer_blocks)
        # Normalizes transformer output (x) using RMS normalization (self.norm)
        x = self.norm(x)  # (bs, seq_len, dim_emb)
        # Projects normalized output (x) to obtain logits over the vocabulary (vocab_size)
        x = self.projection_head(x)  # (bs, seq_len, vocab_size) - represents the logits for each token in the vocab

        # Returns the logits for each token in the vocabulary
        return x  # (bs, seq_len, vocab_size)

    
    # Decorator that ensures model is in inference mode
    Tensor.training = False
    # Method that generates a sequence of tokens using the model given an initial input sequence 
    def generate(
        self, 
        inputs: Tensor, # Initial input sequence tensor
        max_seq_len: int, # Maximum sequence length to generate
        stop_tokens: set | None = None, # Set of tokens to stop generation at
        temperature: float = 0.6, # Controls sampling randomness - higher values increase diversity
        top_p: int = 0.8, # Threshhold parameter for top-p sampling, controlling subset of tokens considered for sampling
    ) -> Tensor:
        for _ in range(max_seq_len):
            # make sure the sequence we're generating doesn't exceed model's sequence length
            inputs_cond = inputs if inputs.size(1) <= self.seq_len else inputs[:, -self.seq_len :]

            # get logits for the last step only, and rescale them to get a probability distribution over the vocabulary
            logits = self(inputs_cond)[:, -1, :]  # (bs, vocab_size)
            probs = (logits / temperature).softmax(axis=-1)  # (bs, vocab_size)

            # sample the next token index using top-p sampling
            next_token = sample_top_p(probs, top_p)  # (bs, 1)

            # stop generation when a stop token is generated
            if stop_tokens is not None and next_token.item() in stop_tokens:
                break

            # append to the sequence being generated
            inputs = Tensor.cat((inputs, next_token), dim=-1)

        # return generated sequence after removing any singleton dimensions and moving it to the CPU
        return inputs.squeeze().cpu()
