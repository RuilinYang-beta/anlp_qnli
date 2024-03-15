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
        'returns: tensor of shape (batch_size,output_size->i.e.number of classes)'
        #Embedding lookup?
        embedded = self.embedding(input)
        embedded_avg = torch.mean(embedded, dim=1)
        out = self.relu(self.fc1(embedded_avg))
        out = self.fc2(out)
        return out
