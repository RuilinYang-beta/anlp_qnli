import time
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from playsound import playsound

from models.SimpleRNN import SimpleRNN
from data_loading.StressDataset import StressDataset
from data_loading.transforms import transform1, target_transform
from data_loading.utils import pad_x_tensors
from evaluation import evaluate_model
from statics import DEVICE, SEED, Notation

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def train(model, isRNN, train_dataloader, optimizer, criterion, **kwargs): 
  """
  Training loop of one epoch.
  Exclude hyperparams as many as possible (they go into `tuner` function).
  Some hyperparams have to be included for the `forward` method of model, eg. `hidden_size` for RNN,
  they are passed as kwargs.
  """
  epoch_loss = 0

  for idx, (x, y, seq_lengths) in enumerate(train_dataloader):   
    assert len(x.size()) == 2, "In training loop, please do batch training; x should be a 2D tensor."

    batch_loss = 0

    x = x.to(DEVICE)
    y = y.to(DEVICE)
    seq_lengths = seq_lengths.to(DEVICE)

    if isRNN: 
      N, _ = x.size()   
      hidden_size = kwargs.get('hidden_size', None)
      h_0 = model.init_hidden((1, N, hidden_size)).to(DEVICE)
      out, h_n = model(x, h_0, seq_lengths)
    else: 
      return None  # TODO: [Ruilin] transformer / FFWD

    batch_loss = criterion(out, y)
    epoch_loss += batch_loss.item()

    batch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

  return epoch_loss



def tuner(dataset, 
          model=SimpleRNN, isRNN=True,  # RNN has diff init way and diff forward method
          # --- for training loop ---
          n_epochs=3, batch_size=300, learning_rate=0.0001,  
          # --- for model shape --- 
          embedding_dim=64,
          output_size=3,
          # --- [RNN only] ---
          hidden_size=128, 
          # --- for optimizer ---
          optimizer=torch.optim.SGD,
          # n_layers=2,   # stacked RNN not ready yet
          # --- for reporting ---
          # print_every=100, plot_every=10, 
          ):
  """
  A wrapper that wraps hyperparameters and pass them to training loop. 
  """
  vocab_size = dataset.get_vocab_size()
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_x_tensors)

  if isRNN: 
    model = model(vocab_size, embedding_dim, hidden_size, output_size).to(DEVICE)
  else: 
    return None  # TODO: [Ruilin] init transformer / FFWD

  optimizer = optimizer(model.parameters(), lr=learning_rate)
  criterion = nn.CrossEntropyLoss(reduction='sum')   # so that batch loss is the sum of all losses of the batch

  start_time = time.time()
  losses_by_epoch = []
  for epoch in range(n_epochs):
    epoch_loss = train(model, isRNN, dataloader, optimizer, criterion, 
                      hidden_size=hidden_size)

    losses_by_epoch.append(epoch_loss)
    # TODO: [Ellie] log this info to a file
    print(f"Epoch-{epoch}, avg loss per example in epoch {epoch_loss / len(dataset)}")

  end_time = time.time()
  elapsed_time = end_time - start_time

  return model, losses_by_epoch, elapsed_time

# TODO: [Ellie/Ruilin] how to make `evaluation` compatible for all occasions? perhaps something look like `train`? 
#  TODO: [Ellie] after `tuner`, evaluate model on dev set
# TODO: [Ellie] after `tuner`, perhaps save the trained model? 
# -> likely helpful for debugging, and for comparing models, and for caputuring the best model
# TODO: [Ellie] inside `tuner`,  log hyperparam info? something like below? 
# -> helpful for comparing models, analyze, reproduce, etc.


# max_length = 15
# print(f"{'Took'.ljust(max_length)}: {elapsed_time} seconds;")
# print(f"{'Device'.ljust(max_length)}: {DEVICE};")
# print(f"{'Epoch'.ljust(max_length)}: {epoch};")
# print(f"{'Learning rate'.ljust(max_length)}: {learning_rate};")
# print(f"{'Batch size'.ljust(max_length)}: {batch_size};")
# print(f"{'Embedding dim'.ljust(max_length)}: {embedding_dim};")
# print(f"{'Hidden size'.ljust(max_length)}: {hidden_size};")

