import time
import json

import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
# from playsound import playsound

from models.SimpleRNN import SimpleRNN
from models.FeedForwardNN import FeedForwardNN
from models.SimpleTransformer import SimpleTransformer
from models.ModelFactory import ModelFactory
from data_loading.StressDataset import StressDataset
from data_loading.transforms import transform1, target_transform
from data_loading.utils import pad_x_tensors
from evaluation import evaluate_model
from statics import DEVICE, SEED, Notation
from utils import _log, _generate_random_int, _generate_random_learning_rate


torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def train(model, train_dataloader, optimizer, criterion,
          ): 
  """
  Training loop of one epoch.
  """

  epoch_loss = 0

  for idx, (x, y, seq_lengths) in enumerate(train_dataloader):   
    assert len(x.size()) == 2, "In training loop, please do batch training; x should be a 2D tensor."

    batch_loss = 0

    x = x.to(DEVICE)
    y = y.to(DEVICE)
    seq_lengths = seq_lengths.to(DEVICE)

    # call model-specific forward method
    if "RNN" in model.__class__.__name__: 
      out, _ = model(x, seq_lengths)

    elif "Transformer" in model.__class__.__name__:
      out = model(x, seq_lengths)

    else: 
      out = model(x)

    batch_loss = criterion(out, y)
    epoch_loss += batch_loss.item()

    optimizer.zero_grad()
    batch_loss.backward()
    # gradient clipping - I don't fully understand yet
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

  return epoch_loss


def tuner(dataset, 
          model_class=SimpleRNN, 
          # --- get from commandline arg ---
          n_epochs=None,
          # --- common hyperparams ---
          batch_size=300, learning_rate=0.0001,  
          embedding_dim=128,
          # --- model specific hyperparams ---
          hidden_size=128,      # for RNN and FFNN
          num_layers=1,         # for RNN and FFNN  
          num_blocks=4,         # for Transformer
          num_heads=4,          # for Transformer
          # --------- other concerns ---------
          log=False, filename=None
          ):
  """
  A wrapper that wraps hyperparameters and pass them to training loop. 
  """
  # ------ fixed hyperparams - we don't have time to experiment ------
  optimizer = torch.optim.SGD   
  output_size = 3
  dropout = 0.2
  # ------------------------------------------------------------------

  vocab_size = dataset.get_vocab_size()
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_x_tensors)

  # encapsulate model creation logic in `ModelFactory`
  model, chosen_hyperparams = ModelFactory.init_model(model_class, 
                              vocab_size, 
                              embedding_dim=embedding_dim,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              num_blocks=num_blocks,
                              num_heads=num_heads,
                              output_size=output_size,
                              dropout=dropout
                            )

  num_params = sum(p.numel() for p in model.parameters())

  hyperparams = {
                  "n_epochs": n_epochs,
                  "batch_size": batch_size, 
                  "learning_rate": learning_rate,
                  "embedding_dim": embedding_dim,
                  **chosen_hyperparams,
                }

  
  if log: 
    _log(filename, json.dumps(hyperparams, indent=4))
    _log(filename, f"Model has {num_params:,} parameters.")
    _log(filename, "-------------------------------")

  print(json.dumps(hyperparams, indent=4))
  print(f"Model has {num_params:,} parameters.")   
  print("-------------------------------")

  optimizer = optimizer(model.parameters(), lr=learning_rate)
  criterion = nn.CrossEntropyLoss(reduction='sum')   # so that batch loss is the sum of all losses of the batch

  start_time = time.time()
  losses_by_epoch = []
  
  for epoch in range(1, n_epochs+1):
    epoch_loss = train(model, 
                      dataloader, 
                      optimizer, 
                      criterion
                      )

    losses_by_epoch.append(epoch_loss)

    if epoch % 10 == 0:
      print(f"Epoch-{epoch}, avg loss per example in epoch {epoch_loss / len(dataset)}")
      if log:
        _log(filename, f"Epoch-{epoch}, avg loss per example in epoch {epoch_loss / len(dataset)}") 

  end_time = time.time()
  elapsed_time = end_time - start_time

  if log: 
    _log(filename, "-------------------------------")
    _log(filename, f"Training took {elapsed_time} seconds.")
    _log(filename, "-------------------------------")

  print(f"Training took {elapsed_time} seconds.")
  print("-------------------------------")

  return model, losses_by_epoch, elapsed_time


def generate_hyperparam_set(): 
  """
  Generate a random hyperparameter set for hyperparameter tuning.
  These are the tune-able hyperparameters that we cherry pick from all possible ones.
  """
  batch_size = _generate_random_int(200, 500, 100)
  learning_rate = _generate_random_learning_rate()

  embedding_dim = _generate_random_int(64, 256, 64)
  hidden_size = _generate_random_int(64, 256, 64)
  num_layers = _generate_random_int(1, 3, 1)
  num_blocks = _generate_random_int(2, 4, 1)
  num_heads = _generate_random_int(2, 4, 1)

  return {
    "learning_rate":    learning_rate, 
    "batch_size":       batch_size, 
    "embedding_dim":    embedding_dim, 
    "hidden_size":      hidden_size, 
    "num_layers":       num_layers, 
    "num_blocks":       num_blocks, 
    "num_heads":        num_heads
    }