import time
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from playsound import playsound

from models.SimpleRNN import SimpleRNN
from models.FeedForwardNN import FeedForwardNN
from models.SimpleTransformer import SimpleTransformer
from data_loading.StressDataset import StressDataset
from data_loading.transforms import transform1, target_transform
from data_loading.utils import pad_x_tensors
from evaluation import evaluate_model
from statics import DEVICE, SEED, Notation

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def train(model, train_dataloader, optimizer, criterion,
          isRNN=False, isFFNN=False, isTransformer=False,
          **kwargs): 
  """
  Training loop of one epoch.
  Exclude hyperparams as many as possible (they go into `tuner` function).
  """

  model_flag = sum([isRNN, isFFNN, isTransformer])
  assert model_flag == 1, "Please specify one and only one model type."

  epoch_loss = 0

  for idx, (x, y, seq_lengths) in enumerate(train_dataloader):   
    assert len(x.size()) == 2, "In training loop, please do batch training; x should be a 2D tensor."

    batch_loss = 0

    x = x.to(DEVICE)
    y = y.to(DEVICE)
    seq_lengths = seq_lengths.to(DEVICE)

    # call model-specific forward method
    if isRNN: 
      out, _ = model(x, seq_lengths)

    if isFFNN:
      out = model(x)

    if isTransformer: 
      out = model(x, seq_lengths)

    batch_loss = criterion(out, y)
    epoch_loss += batch_loss.item()

    batch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

  return epoch_loss


def tuner(dataset, 
          model=SimpleRNN, 
          isRNN=False,  # RNN has diff init way and diff forward method, 
          isFFNN=False,
          isTransformer=False,
          # --- common hyperparams - for training loop ---
          n_epochs=1, batch_size=300, learning_rate=0.0001,  
          optimizer=torch.optim.SGD,
          # --- common hyperparams - for model shape ---
          embedding_dim=128,
          output_size=3,      # let's fix it to be 3, we probably don't have time to experiment 2-binary classification
          dropout=0.2,
          # --- [model specific hyperparams] ---
          hidden_size=128,      # for RNN and FFNN
          num_layers=1,         # for RNN and FFNN  (TODO: enable FFNN to have stacked layers)
          bidirectional=False,  # for RNN
          num_blocks=4,         # for Transformer
          num_heads=4,          # for Transformer
          ):
  """
  A wrapper that wraps hyperparameters and pass them to training loop. 
  """
  model_flag = sum([isRNN, isFFNN, isTransformer])
  assert model_flag == 1, "Please specify one and only one model type."

  vocab_size = dataset.get_vocab_size()
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_x_tensors)

  # initialize models in diff ways
  if isRNN: 
    model = model(vocab_size, embedding_dim, hidden_size, output_size, 
                  dropout=dropout,
                  num_layers=num_layers, bidirectional=bidirectional).to(DEVICE)

  if isFFNN: 
    model = model(vocab_size, embedding_dim, hidden_size, output_size, 
            dropout=dropout, num_layers=num_layers).to(DEVICE)

  if isTransformer: 
    model = model(vocab_size, embedding_dim,
                  num_blocks, num_heads,
                  output_size, 
                  dropout=dropout).to(DEVICE)

  num_params = sum(p.numel() for p in model.parameters())
  print(f"Model has {num_params:,} parameters.")    # would this be sth interesting to log?

  optimizer = optimizer(model.parameters(), lr=learning_rate)
  criterion = nn.CrossEntropyLoss(reduction='sum')   # so that batch loss is the sum of all losses of the batch

  start_time = time.time()
  losses_by_epoch = []
  for epoch in range(n_epochs):
    # pass all hyperparams that `train` possibly need 
    epoch_loss = train(model, dataloader, optimizer, criterion, 
                      isRNN=isRNN, isFFNN=isFFNN, isTransformer=isTransformer,
                      hidden_size=hidden_size, 
                      num_blocks=num_blocks, num_heads=num_heads)

    losses_by_epoch.append(epoch_loss)
    # TODO: [Ellie] log this info to a file
    print(f"Epoch-{epoch}, avg loss per example in epoch {epoch_loss / len(dataset)}")

  end_time = time.time()
  elapsed_time = end_time - start_time

  return model, losses_by_epoch, elapsed_time

#  TODO: [Ellie] after `tuner`, evaluate model on dev set
# TODO: [Ellie] after `tuner`, perhaps save the trained model? 
# -> likely helpful for debugging, and for comparing models, and for caputuring the best model
# TODO: [Ellie] inside `tuner`,  log hyperparam info? something like below? plus the loss per certain epochs? 
# -> helpful for comparing models, analyze, reproduce, etc.


# max_length = 15
# print(f"{'Took'.ljust(max_length)}: {elapsed_time} seconds;")
# print(f"{'Device'.ljust(max_length)}: {DEVICE};")
# print(f"{'Epoch'.ljust(max_length)}: {epoch};")
# print(f"{'Learning rate'.ljust(max_length)}: {learning_rate};")
# print(f"{'Batch size'.ljust(max_length)}: {batch_size};")
# print(f"{'Embedding dim'.ljust(max_length)}: {embedding_dim};")
# print(f"{'Hidden size'.ljust(max_length)}: {hidden_size};")