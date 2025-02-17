import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('..')  # Add the parent directory to the sys.path

from statics import SEED

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

class FeedForwardNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size,
            dropout=0.0, num_layers=1  # ignoring them for now
    ):
        super(FeedForwardNN, self).__init__()
        self.num_layers = num_layers
        #Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        #Linear function
        self.fc1 = nn.Linear(embedding_dim, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # hidden layers 
        self.hidden_layers = nn.ModuleList()
        for i in range(num_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
            self.hidden_layers.append(nn.ReLU())
            self.hidden_layers.append(nn.Dropout(dropout))

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

        x = self.fc1(embedded_avg)
        x = self.relu(x)
        if self.num_layers > 1:
            # only dropout when we have hidden layers
            x = self.dropout(x)

        for layer in self.hidden_layers:
            x = layer(x)

        out = self.fc2(x)
        return out
    
