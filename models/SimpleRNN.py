import torch
import torch.nn as nn

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

import sys
sys.path.append('..')  # Add the parent directory to the sys.path

from statics import SEED

"""
A wrapper of RNN layer in PyTorch, 
with an embedding layer, and a linear layer that maps to the desired output size.
"""

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


# TODO: make use of dropout


class SimpleRNN(nn.Module):

  def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, dropout=0.2,
              num_layers=1, bidirectional=False):
    super(SimpleRNN, self).__init__()
    
    self.D = 2 if bidirectional else 1

    # ----- from indices of one-hot to embedding -----
    # the embeddings at index 0 is for padding token, we don't need to update it 
    # see `data_loading/StressDataset.py` that the token at index 0 of vocab is a padding token
    # see also `data_loading/utils.py` for how we pad the sequences for dataloader batched data
    self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

    # ----- RNN layer -----
    self.rnn = nn.RNN(embedding_dim, hidden_size, 
                      batch_first=True, 
                      num_layers=num_layers, bidirectional=bidirectional, 
                      dropout=dropout)

    # ----- from output of RNN to the real output we want -----
    # need to add softmax layer? -> no, because CrossEntropyLoss does it for us
    self.decoder = nn.Linear(hidden_size * self.D, output_size)  

  def forward(self, x_tensor, seq_lengths=None):
    """
    Forward pass, 
    input (a tensor of indices of tokens of one training example) -> embeddings (matrix of (seq_len, embedding_dim)); 
    embeddings -> feed to built-in RNN -> output of RNN (matrix of (seq_len, hidden_size));
    last output of RNN (hidden_size) -> feed to decoder -> output of the model (output_size)

    `x_tensor`: tensor of (seq_len) or (N, seq_len) where N is batch_size, it's the indices of tokens
    `h0`: tensor of shape (1, N, output_size)

    return: tensor of shape (output_size), h_n
    """
    emb = self.embedding(x_tensor)

    if emb.dim() == 2:    
      # for single example training
      # `x_tensor`:  (L, )
      # `emb`:       (L, embedding_dim)
      # `out`:       (L, hidden_size)
      # `final_out`: (L, output_size) -> (output_size, )
      out, h_n = self.rnn(emb)   # --> h_0 defaults to zero, let it be 
      final_out = self.decoder(out)
      final_out = final_out[-1, :]  

      return final_out, h_n
    elif emb.dim() == 3:  
      # for batch training
      assert seq_lengths is not None, "seq_lengths should not be None for batch training"
      # ------------ no pack_padded_sequence, some computation resources is wasted ---------
      # ------------ supposedly slow but it's fast, somehow ------------
      
      # `x_tensor`:  (N, L)
      # `emb`:        (N, L, embedding_dim)
      # `out`:        (N, L, hidden_size * D) where D is 2 if bidirectional, else 1
      # `out_real`:   (N, hidden_size * D)
      # `out_final`:  (N, output_size)
      out, h_n = self.rnn(emb)

      indices = (seq_lengths - 1)
      out_real = out[torch.arange(out.size(0)), indices, :]

      out_final = self.decoder(out_real)

      return out_final, h_n
      # ------------ with pack_padded_sequence and pad_packed_sequence ---------
      # ------------ supposedly fast but it's slow, somehow ------------
      # # turn embeddings into a PackedSequence

      # # seq_lengths should be on CPU
      # # see https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html
      # seq_lengths = seq_lengths.cpu()  

      # emb = pack_padded_sequence(emb, seq_lengths, 
      #                             batch_first=True
      #                             enforce_sorted=False)

      # out, h_n = self.rnn(emb, h_0)

      # out_unpacked, _ = pad_packed_sequence(out, batch_first=True)
      # # print(f"out_unpacked.size(): {out_unpacked.size()}")

      # indices = (seq_lengths - 1)
      # out_real = out_unpacked[torch.arange(out_unpacked.size(0)), indices, :]
      # # print(f"out_real.size(): {out_real.size()}")

      # out_final = self.decoder(out_real)
      # # print(f"out_final.size(): {out_final.size()}")
      # return out_final, h_n
      # --------------------------------------------------------------------------------
    else: 
      raise ValueError("emb should be 2D or 3D tensor")

  @staticmethod
  def get_type(): 
    return "RNN"