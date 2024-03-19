import time
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
# from playsound import playsound

from training_helper import tuner
from models.FeedForwardNN import FeedForwardNN
from models.SimpleRNN import SimpleRNN
from models.SimpleTransformer import SimpleTransformer
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
# Train a model with default hyperparams in `tuner` function in `training_helper.py`

# --- flags ---
# change the models and flags there, the flags is also shared to evaluation function
# later on we can get these flags from terminal arguments
model = SimpleTransformer
isFFNN = False
isRNN = False
isTransformer = True

# --- model name for saving it ---
model_name = "transformer"    

# --- in case RNN, hidden_size is needed ---
hidden_size=128

# there's more hyperparams in `tuner` function in `training_helper.py`
model, losses_by_epoch, elapsed_time = tuner(dataset=train_set, model=model, 
                                            hidden_size=hidden_size,
                                            isFFNN=isFFNN, isRNN=isRNN, isTransformer=isTransformer)

print(f"{'Took'.ljust(15)}: {elapsed_time} seconds;")

# ================== save model ==================
# Specify the file path where you want to save the model
model_path = f'trained_models/{model_name}.pth'

# Save both model architecture and parameters
torch.save(model, model_path)

# To load the model back later, you can use:
# model = torch.load(model_path)

# playsound('temp_sound/mammal.mp3')  # play sound to notify the end of training


# ================== dummy eval ==================

# TODO: [Ellie] consider encapsulate these two functions into a bigger eval function 
#       and put it inside your hyperparam tuning function,
#       because it has so many common hyperparams with `tuner` function


criterion = nn.CrossEntropyLoss(reduction='sum')

print("------------------ eval the freshly trained model ------------------")

# evaluate on training set -> supposedly performs better than on dev
loss, accuracy, f1_macro = evaluate_model(model, 
                                          train_set,  
                                          criterion,
                                          isFFNN=isFFNN, isRNN=isRNN, isTransformer=isTransformer,
                                          hidden_size=hidden_size
                                          )
print(f"[train] Loss: {loss}, Accuracy: {accuracy}, F1-macro: {f1_macro}")

# evaluate on dev set
dev_set = StressDataset('data/dev.json', Notation.ORIGINAL_CHAR, forEval=True,
                        transform=transform1, 
                        target_transform=target_transform, 
                        vocab=train_set.get_vocab())

loss, accuracy, f1_macro = evaluate_model(model, 
                                          dev_set, 
                                          criterion,
                                          isFFNN=isFFNN, isRNN=isRNN, isTransformer=isTransformer,
                                          hidden_size=hidden_size
                                          )
print(f"[dev]   Loss: {loss}, Accuracy: {accuracy}, F1-macro: {f1_macro}")