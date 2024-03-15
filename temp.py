import time
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from playsound import playsound

from training_helper import tuner
from models.SimpleRNN import SimpleRNN
from data_loading.StressDataset import StressDataset
from data_loading.transforms import transform1, target_transform
from data_loading.utils import pad_x_tensors
from evaluation import evaluate_model
from statics import DEVICE, SEED, Notation

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


# ================== Dataset ==================
train_set = StressDataset('data/train.json', Notation.ORIGINAL_CHAR, forEval=False,
                        transform=transform1, 
                        target_transform=target_transform)

# x = dataset[0][0]
# y = dataset[0][1]

# print((x, y))

# ================== Train with certain hyperparam ==================
# This function will train a model with the below default hyperparams
# see `training_helper.py`
model, losses_by_epoch, elapsed_time = tuner(dataset=train_set)

# def tuner(dataset, 
#           model=SimpleRNN, isRNN=True,  # RNN has diff init way and diff forward method
#           # --- for training loop ---
#           n_epochs=3, batch_size=300, learning_rate=0.0001,  
#           # --- for model shape --- 
#           embedding_dim=64,
#           output_size=3,
#           # --- [RNN only] ---
#           hidden_size=128, 
#           # --- for optimizer ---
#           optimizer=torch.optim.SGD,
#           # n_layers=2,   # stacked RNN not ready yet
#           # --- for reporting ---
#           # print_every=100, plot_every=10, 
#           ):

# # ================== save model ==================
# # Specify the file path where you want to save the model
# model_path = 'trained_models/simple_rnn_b.pth'

# # Save both model architecture and parameters
# torch.save(model, model_path)

# # To load the model back later, you can use:
# # model = torch.load(model_path)

# playsound('temp_sound/mammal.mp3')  # play sound to notify the end of training


# ================== dummy eval ==================

hidden_size=128
criterion = nn.CrossEntropyLoss(reduction='sum')

print("------------------ eval the freshly trained model ------------------")

# evaluate on training set -> supposedly performs better than on dev
loss, accuracy, f1_macro = evaluate_model(model, 
                                          model.init_hidden((1, hidden_size)).to(DEVICE), 
                                          train_set,  # -> change to train_data_set
                                          criterion)
print(f"[train] Loss: {loss}, Accuracy: {accuracy}, F1-macro: {f1_macro}")

# evaluate on dev set
dev_set = StressDataset('data/dev.json', Notation.ORIGINAL_CHAR, forEval=True,
                        transform=transform1, 
                        target_transform=target_transform, 
                        vocab=train_set.get_vocab())

loss, accuracy, f1_macro = evaluate_model(model, 
                                          model.init_hidden((1, hidden_size)).to(DEVICE), 
                                          dev_set, 
                                          criterion)
print(f"[dev]   Loss: {loss}, Accuracy: {accuracy}, F1-macro: {f1_macro}")