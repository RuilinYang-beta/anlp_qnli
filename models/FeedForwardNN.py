import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForwardNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size):
        super(FeedForwardNN, self).__init__()
        #Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        #Linear function
        self.fc1 = nn.Linear(embedding_dim, hidden_size)
        #ReLu
        self.relu = nn.ReLU()
        #Linear function (output)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        # --- for batched data --- (we use it in training to speed up)
        # input:            (N, L)
        # embedded:         (N, L, embedding_dim)
        # embedded_avg:     (N, embedding_dim)
        # out:              (N, output_size)
        # --- for single data --- (we use it for evaluation )
        # input:            (L)
        # embedded:         (L, embedding_dim)
        # embedded_avg:     (embedding_dim)
        # out:              (output_size)
        embedded = self.embedding(input)
        embedded_avg = torch.mean(embedded, dim=-2)  # use relative dimension to handle both single and batched data
        out = self.relu(self.fc1(embedded_avg))
        out = self.fc2(out)
        return out
