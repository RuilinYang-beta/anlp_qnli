"""
Ideally when switching to use a different model, we only need to change 
model = OldModel(..) to model = NewModel(), 
the rest of the code could have minor change but mostly the same, 
especially when BiRNN and SimpleRNN are very similar.

The playground for the two models are separated for now, this saves manual labor 
of commenting and uncommenting 10+ lines of code every time I want to switch model.
"""
import time
import torch 
import torch.nn as nn

from playsound import playsound

from data_loading.StressDataset import StressDataset
from data_loading.transforms import transform1, target_transform
from utils import load_data, pretty_print, build_vocabulary, xy_pair_generator  
from statics import DEVICE, SEED, Notation
from models.SimpleRNN import SimpleRNN
from evaluation import evaluate_model


torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# ================== loading data ==================
train = load_data("train")
# # pretty_print(train[10])

# # get vocab based on dataset and notation
# vocab = build_vocabulary(train, Notation.ORIGINAL_CHAR)

dataset = StressDataset('data/train.json', Notation.ORIGINAL_CHAR, forEval=False,
                        transform=transform1, 
                        target_transform=target_transform)

# ================== quasi-batch ==================
# changing params per `nudge_every`
# difference with real batch training is no parallelism here
nudge_every = 100

# ================== hyperparams / params ==================
epoch = 5
learning_rate = 0.0001

# params for model
embedding_dim = 32    # start with smaller params
hidden_size = 50
output_size = 3

# model 
model = SimpleRNN(dataset.get_vocab_size(), 
                  embedding_dim, hidden_size, output_size).to(DEVICE)
h_0 = model.init_hidden((1, hidden_size)).to(DEVICE)

# for updating model
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# ================== training loop ==================
start_time = time.time()

losses_by_epoch = []

# ----- epoch loop -----
for i in range(epoch):
  # loader = xy_pair_generator(train, Notation.ORIGINAL_CHAR, vocab)

  epoch_loss = 0
  nudge_every_loss = 0

  # for idx, xy_pair in enumerate(loader):
    # x, y = xy_pair
  for idx, (x, y) in enumerate(dataset):    
    x = x.to(DEVICE)
    y = y.to(DEVICE)

    final_out, h_n = model(x, h_0)

    loss = criterion(final_out, y)

    epoch_loss += loss
    nudge_every_loss += loss

    # --------------- nudge params every `nudge_every` examples --------------- #
    if idx % nudge_every == 0:
      # print(f"Epoch-{i}, at example-{idx}, avg loss of past {nudge_every} "
      #       f"example {round(epoch_loss.item() / nudge_every, 2)}")

      model.zero_grad()
      nudge_every_loss.backward()
      optimizer.step()

      nudge_every_loss = 0
    
  print(f"Epoch-{i}, avg loss per example in epoch {epoch_loss.item() / len(train)}")
    # ------------------------------------------------------------------------ #

  # ---------------- nudge params at the end of epoch ------------------------ #
  # losses_by_epoch.append(epoch_loss.item())

  # model.zero_grad()
  # epoch_loss.backward()
  # optimizer.step()

  # print(f"Epoch-{i}, avg loss per example in epoch {epoch_loss.item() / len(train)}")
  # ------------------------------------------------------------------------ #

end_time = time.time()
elapsed_time = end_time - start_time

print(f"------------ unbatched training ------------")
max_length = 15
print(f"{'Took'.ljust(max_length)}: {elapsed_time} seconds;")
print(f"{'Device'.ljust(max_length)}: {DEVICE};")
print(f"{'Epoch'.ljust(max_length)}: {epoch};")
print(f"{'Learning rate'.ljust(max_length)}: {learning_rate};")
print(f"{'nudge every'.ljust(max_length)}: {nudge_every};")
print(f"{'Embedding dim'.ljust(max_length)}: {embedding_dim};")
print(f"{'Hidden size'.ljust(max_length)}: {hidden_size};")

# ------- save model ------- #
# Specify the file path where you want to save the model
model_path = 'trained_models/simple_rnn.pth'

# Save both model architecture and parameters
torch.save(model, model_path)

# To load the model back later, you can use:
# loaded_model = torch.load(model_path)

# `pip install playsound` to install this package
playsound('temp_sound/mammal.mp3')  # play sound to notify the end of training

# ================== dummy eval ==================

print("------------------ eval the freshly trained model ------------------")

# evaluate on training set -> supposedly performs better than on dev
loss, accuracy, f1_macro = evaluate_model(model, 
                                          model.init_hidden((1, hidden_size)).to(DEVICE), 
                                          dataset,  
                                          criterion)
print(f"[train] Loss: {loss}, Accuracy: {accuracy}, F1-macro: {f1_macro}")

# evaluate on dev set
devset = StressDataset('data/dev.json', Notation.ORIGINAL_CHAR, forEval=True,
                        transform=transform1, 
                        target_transform=target_transform,
                        vocab=dataset.get_vocab())

loss, accuracy, f1_macro = evaluate_model(model, 
                                          model.init_hidden((1, hidden_size)).to(DEVICE), 
                                          devset, 
                                          criterion)
print(f"[dev]   Loss: {loss}, Accuracy: {accuracy}, F1-macro: {f1_macro}")

