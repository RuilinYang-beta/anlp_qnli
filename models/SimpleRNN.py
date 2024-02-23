import torch
import torch.nn as nn


"""
We want more control over the RNN than what PyTorch's RNN class provides. For two reasons: 

- The RNN impl in PyTorch defaults output size to be the same as hidden size (or double the hidden size if bidirectional=True).
  https://pytorch.org/docs/stable/generated/torch.nn.RNN.html#torch.nn.RNN
  But we want to be able to control the output size: it's either 3 for 3-class classification, or 2 if we build two binary classifiers. 

- We want to map the input to an embedding as an abstraction of the input, and we want different way to do this: 
  For non-numeral tokens, we want to map them to an embedding;
  For numeral tokens, we have different ways to map them to an embedding, depending on the notation of numbers. 

"""
class SimpleRNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        # self.num_layers = num_layers    # suspend num_layers for now


        # ===== from indices of one-hot to embedding =====
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # ===== RNN layer =====
        # off-the-shelf RNN layer, see https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        """
        torch.nn.RNN(self, input_size, hidden_size, num_layers=1, nonlinearity='tanh', bias=True, batch_first=False, dropout=0.0, bidirectional=False, device=None, dtype=None)
        """
        self.rnn = nn.RNN(embedding_dim, hidden_size)

        # ===== from output of RNN to the real output we want =====
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, h_0):
        """
        Forward pass, 
        input (a tensor of indices of tokens of one training example) -> embeddings (matrix of (seq_len, embedding_dim)); 
        embeddings -> feed to built-in RNN -> output of RNN (matrix of (seq_len, hidden_size));
        last output of RNN (hidden_size) -> feed to decoder -> output of the model (output_size)

        `input`: tensor of (seq_len), it's the indices of tokens
        `h0`: tensor of shape (1, output_size)

        return: tensor of shape (output_size), h_n
        """
        # print(f"h_0.size(): {h_0.size()}")
        # print(f"input.size(): {input.size()}")

        emb = self.embedding(input)
        # print(f"emb.size(): {emb.size()}")

        out, h_n = self.rnn(emb, h_0)
        # print(f"out.size(): {out.size()}")

        final_out = self.decoder(out[-1, :])
        # print(f"final_out.size(): {final_out.size()}")

        return final_out, h_n

    def init_hidden(self): 
      return torch.zeros(self.hidden_size).unsqueeze(0) # shape (1, hidden_size)


