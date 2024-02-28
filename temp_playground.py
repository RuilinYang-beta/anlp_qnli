"""
Playground to mess with simple RNN. Later it could be part of `qnli.py`.
"""
import torch 
import torch.nn as nn

from utils import load_data, pretty_print, build_vocabulary, xy_pair_generator  
from statics import Notation, Token, notation2key, labels

from models.SimpleRNN import SimpleRNN
from models.BiRNN import BiRNN

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
simpe_model = SimpleRNN(len(vocab), embedding_dim, hidden_size, output_size)

bi_rnn_model = BiRNN(len(vocab), embedding_dim, hidden_size, output_size)

h_0_simple = simpe_model.init_hidden()
h_0_birnn = bi_rnn_model.init_hidden()


# for updating model
optimizer_simplernn = torch.optim.SGD(simpe_model.parameters(), lr=learning_rate)
optimizer_bi_rnn = torch.optim.SGD(bi_rnn_model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
nudge_every = 300

for i in range(epoch):
  loader = xy_pair_generator(train, Notation.ORIGINAL_CHAR, vocab)

  loss_simple = 0
  loss_bi_rnn = 0

  for idx, xy_pair in enumerate(loader):
    x, y = xy_pair
    final_out_simple, h_n = simpe_model(x, h_0_simple)
    final_out_bi_rnn = bi_rnn_model(x, h_0_birnn)

    loss_simple += criterion(final_out_simple, y)
    loss_bi_rnn += criterion(final_out_bi_rnn, y)

    if idx % nudge_every == 0:
      print(f"Epoch-{i}, at example-{idx}, avg loss of past {nudge_every} "
            f"example {round(loss_simple.item() / nudge_every, 2)}")
      print(f"Epoch-{i}, at example-{idx}, avg loss of past {nudge_every} "
            f"examples (BiRNN): {round(loss_bi_rnn.item() / nudge_every, 2)}")

      loss_simple.backward()
      loss_bi_rnn.backward()

      optimizer_simplernn.step()
      optimizer_bi_rnn.step()

      loss_simple = 0
      loss_bi_rnn = 0

      simpe_model.zero_grad()
      bi_rnn_model.zero_grad()
