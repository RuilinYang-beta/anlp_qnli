import torch
import torch.nn as nn

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

import sys
sys.path.append('..')  # Add the parent directory to the sys.path

from statics import SEED

"""
We want more control over the RNN than what PyTorch's RNN class provides. For two reasons: 

- The RNN impl in PyTorch defaults output size to be the same as hidden size (or double the hidden size if bidirectional=True).
  https://pytorch.org/docs/stable/generated/torch.nn.RNN.html#torch.nn.RNN
  But we want to be able to control the output size: it's either 3 for 3-class classification, or 2 if we build two binary classifiers. 

- We want to map the input to an embedding as an abstraction of the input, and we want different way to do this: 
  For non-numeral tokens, we want to map them to an embedding;
  For numeral tokens, we have different ways to map them to an embedding, depending on the notation of numbers. 

"""

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

class SimpleRNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        # self.num_layers = num_layers    # suspend num_layers for now


        # ===== from indices of one-hot to embedding =====
        # the embeddings at index 0 is for padding token, we don't need to update it 
        # see `data_loading/StressDataset.py` that the token at index 0 of vocab is a padding token
        # see also `data_loading/utils.py` for how we pad the sequences for dataloader batched data
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # ===== RNN layer =====
        # off-the-shelf RNN layer, see https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        """
        torch.nn.RNN(self, input_size, hidden_size, num_layers=1, nonlinearity='tanh', bias=True, batch_first=False, dropout=0.0, bidirectional=False, device=None, dtype=None)
        """
        self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=True)

        # ===== from output of RNN to the real output we want =====
        # need to add softmax layer? -> no, because CrossEntropyLoss does it for us
        self.decoder = nn.Linear(hidden_size, output_size)  

    def forward(self, x_tensor, h_0, seq_lengths=None):
        """
        Forward pass, 
        input (a tensor of indices of tokens of one training example) -> embeddings (matrix of (seq_len, embedding_dim)); 
        embeddings -> feed to built-in RNN -> output of RNN (matrix of (seq_len, hidden_size));
        last output of RNN (hidden_size) -> feed to decoder -> output of the model (output_size)

        `x_tensor`: tensor of (seq_len) or (N, seq_len) where N is batch_size, it's the indices of tokens
        `h0`: tensor of shape (1, N, output_size)

        return: tensor of shape (output_size), h_n
        """
        # print(f"h_0.size(): {h_0.size()}")
        # print(f"x_tensor.size(): {x_tensor.size()}")
        # print(f"seq_lengths: {seq_lengths}")

        emb = self.embedding(x_tensor)
        # print(f"emb.size(): {emb.size()}")

        if emb.dim() == 2:    # for single example training, (L, embedding_dim)
          # `x_tensor`:  (L, )
          # `h_0`:  (1, hidden_size)
          # `emb`:  (L, embedding_dim)
          # `out`:  (L, hidden_size)
          # `final_out`:  (output_size, )
          out, h_n = self.rnn(emb, h_0)
          # # print(f"out.size(): {out.size()}")

          final_out = self.decoder(out)
          # print(f"final_out.size() before slicing: {final_out.size()}")

          final_out = final_out[-1, :]  

          # print(f"final_out.size() after slicing: {final_out.size()}")

          return final_out, h_n
        elif emb.dim() == 3:  # for batch training, (N, L, embedding_dim)
          assert seq_lengths is not None, "seq_lengths should not be None for batch training"
          # ------------ no pack_padded_sequence, some computation resources is wasted ---------
          # ------------ supposedly slow but it's fast, somehow ------------
          # `x_tensor`:  (N, L)
          # `h_0`:  (1, N, hidden_size)
          # `emb`:  (N, L, embedding_dim)
          # `out`:  (N, L, hidden_size)
          # `out_real`:  (N, hidden_size)
          # `out_final`:  (N, output_size)
          out, h_n = self.rnn(emb, h_0)
          # print(f"out.size(): {out.size()}")

          indices = (seq_lengths - 1)
          out_real = out[torch.arange(out.size(0)), indices, :]

          out_final = self.decoder(out_real)
          # print(f"final_out.size() before slicing: {final_out.size()}")
          
          # final_out = final_out[:, -1, :] 

          return out_final, h_n
          # --------------------------------------------------------------------------------
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


    # model is agnostic to whether batch training or not
    # so init hidden to what size should be decoupled from model
    def init_hidden(self, size):
      """
      Args:
      - size: a tuple of the size of h_0, 
              (1, batch_size, hidden_size) for batch training, or
              (1, hidden_size) for single example training
      """ 
      return torch.zeros(size) # .unsqueeze(0).unsqueeze(0) # shape (1, hidden_size)


