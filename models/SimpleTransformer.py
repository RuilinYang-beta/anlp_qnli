import torch
import torch.nn as nn
import sys
sys.path.append('..')  # Add the parent directory to the sys.path

from torch.nn import functional as F
from .CustomSequential import CustomSequential
from statics import SEED

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

MAX_SEQ_LENGTH = 512  # max sequence length for positional embeddings

class PositionalEmbeddings(nn.Module):
  """
  Positional embeddings for transformer model, 
  when a token is out of range, return the last positional embedding.
  """
  def __init__(self, max_seq_length, embedding_dim):
    super(PositionalEmbeddings, self).__init__()
    self.max_seq_length = max_seq_length
    self.embedding = nn.Embedding(max_seq_length, embedding_dim)

  def forward(self, input):
    # Clip input indices to be within range [0, num_embeddings-1]
    input_clipped = torch.clamp(input, max=self.max_seq_length-1)
    
    emb = self.embedding(input_clipped)
    return emb

class Head(nn.Module): 
  """
  Compute single head self-attention. 
  For batched input, return the output with padding tokens zeroed out.
  """
  def __init__(self, embedding_dim, head_size):
    super(Head, self).__init__()
    self.key = nn.Linear(embedding_dim, head_size, bias=False)
    self.query = nn.Linear(embedding_dim, head_size, bias=False)
    self.value = nn.Linear(embedding_dim, head_size, bias=False)

  def forward(self, input, seq_lengths=None):
    # N is batch size, L is the longest sequence length of this batch
    # input: (N, L, embedding_dim)  |  (L, embedding_dim)
    # q, k, v: (N, L, head_size)    |  (L, head_size)
    # output: (N, L, head_size)     |  (L, head_size)

    q = self.query(input)
    k = self.key(input)
    v = self.value(input)

    if input.dim() == 2: 

      weights = torch.mm(q, k.t()) / (k.size(-1) ** 0.5)  # (L, L) 
      weights = F.softmax(weights, dim=-1)                # (L, L)
      out = torch.mm(weights, v)                          # (L, head_size)
      return out

    elif input.dim() == 3:  # for batch training, (N, L, embedding_dim)

      # mask out the padding tokens to compute correct attention weights
      indices = torch.arange(input.size(1)).unsqueeze(0)           # (1, L)
      indices = indices.expand(input.size(0), -1).to(input.device) # (N, L)

      mask = indices >= seq_lengths.unsqueeze(1)                   # (N, L)
      mask = mask.unsqueeze(-1)                                    # (N, L, 1)

      q = q.masked_fill(mask, 0)
      k = k.masked_fill(mask, 0)

      weights = torch.bmm(q, k.transpose(1, 2)) / (k.size(-1) ** 0.5)  # (N, L, L) with 0s
      weights = F.softmax(weights, dim=-1)          # (N, L, L) with NaNs
      weights = torch.where(torch.isnan(weights), torch.zeros_like(weights), weights)  # (N, L, L) with 0s

      out = torch.bmm(weights, v)  # (N, L, head_size) with 0s

      return out


class MultiHead(nn.Module):
  """
  Multi-head self-attention, glue together the outputs of multiple heads,
  and project the output to the original embedding dimension.
  """
  def __init__(self, embedding_dim, num_heads, head_size, dropout=0.2):
    super().__init__()
    self.heads = nn.ModuleList([Head(embedding_dim, head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(head_size * num_heads, embedding_dim)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x_tensors, seq_lengths=None):
    # x_tensors:                (N, L, embedding_dim)     |  (L, embedding_dim) 
    # h(x_tensors, seq_length): (N, L, head_size)
    # out:                      (N, L, head_size * num_heads)
    # final_out:                (N, L, embedding_dim) with 0s for padding tokens
    out = torch.cat([h(x_tensors, seq_lengths) for h in self.heads], dim=-1)
    final_out = self.dropout(self.proj(out))
    return final_out

class FeedFoward(nn.Module):
  """ a simple linear layer followed by a non-linearity """

  def __init__(self, embedding_dim, dropout=0.2):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(embedding_dim, 4 * embedding_dim),
        nn.ReLU(),
        nn.Linear(4 * embedding_dim, embedding_dim),
        nn.Dropout(dropout),
    )

  def forward(self, x):
    return self.net(x)

class Block(nn.Module):
  """ 
  Transformer block: 
  - normalization
  - multi-head attention
  - normalization
  - feedforward

  Args: 
  - `embedding_dim`: the dimension of the input embeddings
  - `num_heads`: the number of heads to use in the multi-head attention
  """

  def __init__(self, embedding_dim, num_heads, dropout=0.2):
    super().__init__()
    head_size = embedding_dim // num_heads
    self.sa = MultiHead(embedding_dim, num_heads, head_size, dropout)
    self.ffwd = FeedFoward(embedding_dim, dropout)
    # normalization will add back noise for the padding tokens
    self.ln1 = nn.LayerNorm(embedding_dim) 
    self.ln2 = nn.LayerNorm(embedding_dim)

  def forward(self, x_tensors, seq_lengths):
    # batch training:           x_tensors, normed, after_attention, after_ffwd: (N, L, embedding_dim)
    # single example training:  x_tensors, normed, after_attention, after_ffwd: (L, embedding_dim)
    normed = self.ln1(x_tensors)
    after_attention = normed + self.sa(normed, seq_lengths)
    
    normed = self.ln2(after_attention)
    after_ffwd = normed + self.ffwd(normed)
    return after_ffwd, seq_lengths


class SimpleTransformer(nn.Module): 
  def __init__(self, vocab_size, embedding_dim, num_blocks, num_heads, output_size, dropout=0.2):
    super(SimpleTransformer, self).__init__()
    # token at index 0 of vocab is padding token
    # see `data_loading/StressDataset.py` and `data_loading/utils.py` 
    self.token_embedding_table = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)  
    self.positional_embedding_table = PositionalEmbeddings(MAX_SEQ_LENGTH, embedding_dim)
    self.blocks = CustomSequential(*[Block(embedding_dim, num_heads, dropout) for _ in range(num_blocks)])
    self.ln_f = nn.LayerNorm(embedding_dim) # final layer norm
    self.decode = nn.Linear(embedding_dim, output_size)

  def forward(self, input, seq_lengths=None):
    if input.dim() == 2:
      assert seq_lengths is not None, "seq_lengths should not be None for batch training"

    # N is batch size, L is the longest sequence length of this batch
    # batch training  |  single example training (eval only)

    # input: (N, L)          |  (L)
    # (N, L, embedding_dim)  |  (L, embedding_dim)  
    tok_emb = self.token_embedding_table(input)           
    # (L, embedding_dim)     
    pos_emb = self.positional_embedding_table(torch.arange(input.size(-1)).to(input.device))  
    # (N, L, embedding_dim)  |  (L, embedding_dim)
    x_tensors = tok_emb + pos_emb
    # (N, L, embedding_dim)  |  (L, embedding_dim)
    after_blocks, _ = self.blocks(x_tensors, seq_lengths)    
    # (N, L, embedding_dim)  |  (L, embedding_dim)
    after_norm = self.ln_f(after_blocks)
    # (N, L, output_size)    |  (L, output_size)
    output = self.decode(after_norm)

    if output.dim() == 2: 
      final_out = torch.sum(output, dim=0) / output.size(0)  # (output_size)
      return final_out
      
    elif output.dim() == 3:  # for batch training, (N, L, embedding_dim)

      assert seq_lengths is not None, "seq_lengths should not be None for batch training"

      # mask out the padding tokens
      indices = torch.arange(output.size(1)).unsqueeze(0)            # (1, L)
      indices = indices.expand(output.size(0), -1).to(output.device) # (N, L)

      mask = indices >= seq_lengths.unsqueeze(1)                   # (N, L)
      mask = mask.unsqueeze(-1)                                    # (N, L, 1)

      output = output.masked_fill(mask, 0)   # (N, L, output_size)

      final_out = torch.sum(output, dim=1) / seq_lengths.unsqueeze(1)  # (N, output_size)

      return final_out  
    else: 
      assert False, "input should be either 2D or 3D tensor."
