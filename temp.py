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

# ================== Dataset ==================
dataset = StressDataset('data/train.json', Notation.ORIGINAL_CHAR, forEval=False,
                        transform=transform1, 
                        target_transform=target_transform)

print(f"dataset size: {len(dataset)}")
print(f"vocab size: {dataset.get_vocab_size()}")
# x = dataset[0][0]
# y = dataset[0][1]

# print((x, y))


# ================== Dataloader, load sample of diff len into batches ==================
nudge_every = 300

batch_size = 300   # will be used to init Dataloader and h_0
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_x_tensors)

# for idx, sample in enumerate(train_dataloader):
#   x, y = sample
  
#   # print(x)
#   # print(f"x.size(): {x.size()}, y.size(): {y.size()}")
#   if idx == 0:
#     break

# ================== hyperparams / params ==================

# hyper params for training loop
epoch = 500
learning_rate = 0.0001

# hyper params for model
embedding_dim = 32
hidden_size = 50
output_size = 3

# model 
model = SimpleRNN(dataset.get_vocab_size(), 
                  embedding_dim, hidden_size, output_size).to(DEVICE)

# for updating model
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(reduction='sum')   # so that batch loss is the sum of all losses of the batch

# ================== training loop ==================
start_time = time.time()

losses_by_epoch = []

# ----- epoch loop -----
for i in range(epoch):
  epoch_loss = 0
  nudge_every_loss = 0

  # --- [start] batch loop w. dataloader ---
  for idx, (x, y, seq_lengths) in enumerate(train_dataloader):   
    assert len(x.size()) == 2, "In training loop, please do batch training; x should be a 2D tensor."

    batch_loss = 0

    x = x.to(DEVICE)
    y = y.to(DEVICE)
    seq_lengths = seq_lengths.to(DEVICE)

    # N is actual batch size, the last batch may be smaller than batch_size
    N, _ = x.size()   
    h_0 = model.init_hidden((1, N, hidden_size)).to(DEVICE)

    # print(f"x: {x}")
    # print(f"x.size(): {x.size()}, y.size(): {y.size()}")

    out_final, h_n = model(x, h_0, seq_lengths)
    # print(f"[in temp] Epoch {i}, out_final: {out_final}")
    # print(f"[in temp] out_final.size(): {out_final.size()}")

    batch_loss = criterion(out_final, y)
    # print(f"[in temp] batch_loss: {batch_loss.item()}")
    epoch_loss += batch_loss

    # update model 
    batch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # print(f"Epoch-{i}, batch-{idx}, avg loss per example in batch: {batch_loss.item() / batch_size}")
  # --- [end] batch loop w. dataloader ---

  # --- [start] batch loop w. dataset -> will be deprecated ---
  # for idx, (x, y) in enumerate(dataset):            # for single example training -> will be deprecated

  #   x = x.to(DEVICE)
  #   y = y.to(DEVICE)

  #   # init h_0
  #   h_0 = model.init_hidden((1, hidden_size)).to(DEVICE)

  #   # print(f"x: {x}")
  #   # print(f"x.size(): {x.size()}, y.size(): {y.size()}")

  #   final_out, h_n = model(x, h_0)

  #   # print(f"[in temp] final_out: {final_out}")
  #   # print(f"[in temp] final_out.size(): {final_out.size()}")
  #   loss = criterion(final_out, y)
  #   nudge_every_loss += loss
  #   epoch_loss += loss
  #   # print(f"[in temp] batch_loss: {batch_loss.item()}")

  #   # break

  #   # ------- for single example training, nudge_every ------- -> will be deprecated
  #   # if idx % nudge_every == 0:
  #   #   nudge_every_loss.backward()
  #   #   optimizer.step()
  #   #   optimizer.zero_grad()

  #   #   epoch_loss += nudge_every_loss.item()
  #   #   nudge_every_loss = 0

  #   #   # print(f"Epoch: {i}, example idx: {idx}, loss: {batch_loss.item()}")
  #   # -----------------------------------------------------------------
  # # ------- for single example training, nudge per epoch ------- -> will be deprecated
  # update model
  # epoch_loss.backward()
  # optimizer.step()
  # optimizer.zero_grad()
  # --- [end] batch loop w. dataset -> will be deprecated ---

  losses_by_epoch.append(epoch_loss.item())
  print(f"Epoch-{i}, avg loss per example in epoch {epoch_loss / len(dataset)}")

end_time = time.time()
elapsed_time = end_time - start_time

print(f"------------ batch training with no pack_pad, pad_pack; supposedly slow, but fast ------------")
# print(f"------------ with pack_pad, pad_pack; supposedly fast, but slow ------------")
max_length = 15
print(f"{'Took'.ljust(max_length)}: {elapsed_time} seconds;")
print(f"{'Device'.ljust(max_length)}: {DEVICE};")
print(f"{'Epoch'.ljust(max_length)}: {epoch};")
print(f"{'Learning rate'.ljust(max_length)}: {learning_rate};")
print(f"{'Batch size'.ljust(max_length)}: {batch_size};")
print(f"{'Embedding dim'.ljust(max_length)}: {embedding_dim};")
print(f"{'Hidden size'.ljust(max_length)}: {hidden_size};")

# ================== save model ==================
# Specify the file path where you want to save the model
model_path = 'trained_models/simple_rnn_b.pth'

# Save both model architecture and parameters
torch.save(model, model_path)

# To load the model back later, you can use:
# model = torch.load(model_path)

playsound('temp_sound/mammal.mp3')  # play sound to notify the end of training

# ================== dummy eval ==================

print("------------------ eval the freshly trained model ------------------")

# evaluate on training set -> supposedly performs better than on dev
loss, accuracy, f1_macro = evaluate_model(model, 
                                          model.init_hidden((1, hidden_size)).to(DEVICE), 
                                          dataset,  # -> change to train_data_set
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