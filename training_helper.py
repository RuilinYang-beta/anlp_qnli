import time
import random
import math
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from playsound import playsound

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
  Exclude hyperparams as many as possible (they go into `tuner` function).
  """

  epoch_loss = 0

  for idx, (x, y, seq_lengths) in enumerate(train_dataloader):   
    assert len(x.size()) == 2, "In training loop, please do batch training; x should be a 2D tensor."

    batch_loss = 0

    x = x.to(DEVICE)
    y = y.to(DEVICE)
    seq_lengths = seq_lengths.to(DEVICE)

    # call model-specific forward method
    if model.get_type() == "RNN": 
      out, _ = model(x, seq_lengths)

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
          # --- common hyperparams - for training loop ---
          n_epochs=1, batch_size=300, learning_rate=0.0001,  
          optimizer=torch.optim.SGD,
          # --- common hyperparams - for model shape ---
          embedding_dim=128,
          output_size=3,      # let's fix it to be 3, we probably don't have time to experiment 2-binary classification
          dropout=0.2,        # let's fix it to be 0.2, we probably don't have time to experiment
          # --- [model specific hyperparams] ---
          hidden_size=128,      # for RNN and FFNN
          num_layers=1,         # for RNN and FFNN  (TODO: enable FFNN to have stacked layers)
          bidirectional=False,  # for BiRNN
          num_blocks=4,         # for Transformer
          num_heads=4,          # for Transformer
          ):
  """
  A wrapper that wraps hyperparameters and pass them to training loop. 
  """

  vocab_size = dataset.get_vocab_size()
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_x_tensors)

  # encapsulate model creation logic in `ModelFactory`
  model = ModelFactory.init_model(model_class, 
                      vocab_size, 
                      embedding_dim=embedding_dim,
                      output_size=output_size,
                      dropout=dropout,
                      hidden_size=hidden_size,
                      num_layers=num_layers,
                      bidirectional=bidirectional,
                      num_blocks=num_blocks,
                      num_heads=num_heads
                      )

  num_params = sum(p.numel() for p in model.parameters())
  print(f"Model has {num_params:,} parameters.")    # would this be sth interesting to log?

  optimizer = optimizer(model.parameters(), lr=learning_rate)
  criterion = nn.CrossEntropyLoss(reduction='sum')   # so that batch loss is the sum of all losses of the batch

  start_time = time.time()
  losses_by_epoch = []
  for epoch in range(n_epochs):
    epoch_loss = train(model, 
                      dataloader, 
                      optimizer, 
                      criterion
                      )

    losses_by_epoch.append(epoch_loss)
    print(f"Epoch-{epoch}, avg loss per example in epoch {epoch_loss / len(dataset)}")

  end_time = time.time()
  elapsed_time = end_time - start_time

  return model, losses_by_epoch, elapsed_time


# def custom_train(dataset, model_class):
#   hyperparam_sets = [generate_hyperparam_set() for i in range(3)]


#   for hype in hyperparam_sets: 
#     model, losses_by_epoch, elapsed_time = tuner(dataset, model_class, **hype)

#   print(f"you have successfully reach here!!")


def generate_hyperparam_set(): 
  batch_size = _generate_random_int(64, 512, 64)
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




# max_length = 15
# print(f"{'Took'.ljust(max_length)}: {elapsed_time} seconds;")
# print(f"{'Device'.ljust(max_length)}: {DEVICE};")
# print(f"{'Epoch'.ljust(max_length)}: {epoch};")
# print(f"{'Learning rate'.ljust(max_length)}: {learning_rate};")
# print(f"{'Batch size'.ljust(max_length)}: {batch_size};")
# print(f"{'Embedding dim'.ljust(max_length)}: {embedding_dim};")
# print(f"{'Hidden size'.ljust(max_length)}: {hidden_size};")