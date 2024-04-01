import torch
import torch.nn as nn

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

import sys
sys.path.append('..')  # Add the parent directory to the sys.path

from statics import SEED

"""
Copy and pasted from SimpleRNN, with `bidirectioal=True`. 
Purpose of this class is to make initializing and training more consistent.
"""

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

class BiRNN(nn.Module):

  def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, 
              dropout=0.2,
              num_layers=1
              ):
    super(BiRNN, self).__init__()
    
    self.D = 2 

    # ----- from indices of one-hot to embedding -----
    # the embeddings at index 0 is for padding token, we don't need to update it 
    # see `data_loading/StressDataset.py` that the token at index 0 of vocab is a padding token
    # see also `data_loading/utils.py` for how we pad the sequences for dataloader batched data
    self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

    # ----- RNN layer -----
    self.rnn = nn.RNN(embedding_dim, hidden_size, 
                      batch_first=True, 
                      num_layers=num_layers, bidirectional=True, 
                      dropout=0.0 if num_layers == 1 else dropout)

    # ----- from output of RNN to the real output we want -----
    # need to add softmax layer? -> no, because CrossEntropyLoss does it for us
    self.decoder = nn.Linear(hidden_size * self.D, output_size)  

  def forward(self, x_tensor, seq_lengths=None):
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
    else: 
      raise ValueError("emb should be 2D or 3D tensor")
