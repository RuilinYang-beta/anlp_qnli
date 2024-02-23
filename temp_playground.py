"""
Playground to mess with simple RNN. Later it could be part of `qnli.py`.
"""
import torch 
import torch.nn as nn

from utils import load_data, pretty_print, build_vocabulary, xy_pair_generator  
from statics import Notation, Token, notation2key, labels

from models.SimpleRNN import SimpleRNN

# load data 
train = load_data("train")
# dev = load_data("dev")
# test = load_data("test")

# pretty_print(train[10])

# get vocab based on dataset and notation
vocab = build_vocabulary(train, Notation.ORIGINAL_CHAR)

# hyperparams for training loop 
epoch = 10
learning_rate = 0.01

# params for model
embedding_dim = 128
hidden_size = 256
output_size = 3

# model 
model = SimpleRNN(len(vocab), embedding_dim, hidden_size, output_size)
h_0 = model.init_hidden()


# for updating model
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
nudge_every = 300

for i in range(epoch):
  loader = xy_pair_generator(train, Notation.ORIGINAL_CHAR, vocab)

  loss = 0

  for idx, xy_pair in enumerate(loader):
    x, y = xy_pair
    final_out, h_n = model(x, h_0)

    loss += criterion(final_out, y)

    if idx % nudge_every == 0:
      print(f"Epoch-{i}, at example-{idx}, avg loss of past {nudge_every} example {round(loss.item() / nudge_every, 2)}")

      loss.backward()
      optimizer.step()

      loss = 0 
      model.zero_grad()

