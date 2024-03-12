import torch
import torch.nn as nn

class BiRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, num_layers=1):
        super(BiRNN, self).__init__()
        self.input_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.RNN(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True)

        # multiply by 2 for bidirectional
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input, h_0):
        embedded = self.embedding(input)
        output, h_n = self.rnn(embedded, h_0)
        hidden_cat = torch.flatten(h_n)

        output = self.fc(hidden_cat)
        return output, h_n

    def init_hidden(self):
        return torch.zeros(2, self.hidden_size)
