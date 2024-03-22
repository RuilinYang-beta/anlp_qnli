import time
import random
import math
import json
import os

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

    batch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

  return epoch_loss


def tuner(dataset, 
          model_class=SimpleRNN, 
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
  n_epochs = 1   
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
                  "model_class": model_class.__name__,
                  "n_epochs": n_epochs,
                  "batch_size": batch_size, 
                  "learning_rate": learning_rate,
                  "embedding_dim": embedding_dim,
                  **chosen_hyperparams,
                }

  print(json.dumps(hyperparams, indent=4))
  
  if log: 
    _add_log(filename, "-------------------------------")
    _add_log(filename, json.dumps(hyperparams, indent=4))
    _add_log(filename, "-------------------------------")
    _add_log(filename, f"Model has {num_params:,} parameters.")
    _add_log(filename, "-------------------------------")

  print(f"Model has {num_params:,} parameters.")   

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

    if log: 
      _add_log(filename, f"Epoch-{epoch}, avg loss per example in epoch {epoch_loss / len(dataset)}") 
      # if epoch % 10 == 0:
      #   _add_log(filename, f"Epoch-{epoch}, avg loss per example in epoch {epoch_loss / len(dataset)}") 

  end_time = time.time()
  elapsed_time = end_time - start_time

  if log: 
    _add_log(filename, "-------------------------------")
    _add_log(filename, f"Training took {elapsed_time} seconds.")
    _add_log(filename, "-------------------------------")

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


def _generate_random_learning_rate(lower_bound=0.0001, upper_bound=0.1):
  """
  Return a random learning rate in range (0.0001, 0.1)
  """
  lower = math.log10(lower_bound)
  upper = math.log10(upper_bound)

  r = random.uniform(lower, upper)
  return 10 ** r


def _generate_random_int(min_val, max_val, step):
  """
  Return a random integer in range [min_value, max_value], 
  incremented by step.
  """
  if min_val > max_val:
      raise ValueError("min_value must be less than or equal to max_value")

  if step <= 0:
      raise ValueError("step must be a positive integer")

  num_values = (max_val - min_val) // step + 1
  random_index = random.randint(0, num_values - 1)
  random_value = min_val + random_index * step

  return random_value


def _add_log(filename, message):

  os.makedirs(os.path.dirname(filename), exist_ok=True)

  with open(filename, "a") as f:
    f.write(message + "\n")
