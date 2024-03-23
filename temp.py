import time
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
# from playsound import playsound

from data_loading.StressDataset import StressDataset
from data_loading.transforms import transform1, target_transform

from training_helper import tuner, generate_hyperparam_set, _log
from models.ModelFactory import ModelFactory
from models.FeedForwardNN import FeedForwardNN
from models.SimpleRNN import SimpleRNN
from models.BiRNN import BiRNN
from models.SimpleTransformer import SimpleTransformer

from evaluation import evaluate_model
from statics import DEVICE, SEED, Notation

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# ======= you only need to modify here =======
# --- for data --- 
TRAIN_PATH = 'data/train.json'
NOTATION = Notation.ORIGINAL_CHAR
# --- for model ---
model_class = FeedForwardNN
# --- for log / save model ---
save_model = True
log = True
# ============================================

# --- prepare for training ---
train_set = StressDataset(TRAIN_PATH, NOTATION, forEval=False,
                        transform=transform1, 
                        target_transform=target_transform)

dev_set = StressDataset('data/dev.json', NOTATION, forEval=True,
                        transform=transform1, 
                        target_transform=target_transform, 
                        vocab=train_set.get_vocab())

criterion = nn.CrossEntropyLoss(reduction='sum')

# --- prepare for log/save model  ---
t = "train" if TRAIN_PATH == 'data/train.json' else "train_aug"
n = NOTATION.value
m = model_class.__name__

# --- train models with diff hyperparam sets ---
# --- supported hyperparams ---
# * batch_size
# * learning_rate
# * embedding_dim
# * hidden_size       only FFNN / RNN
# * num_layers        only FFNN / RNN
# * num_blocks        only Transformer 
# * num_heads         only Transformer

# === optionally you can change these here ===
# but eventually we let generate_hyperparam_set to handle it
# I only keep it here because my computer can't handle too complex model
# hyperparam_sets = [ { 
#         'batch_size': 300, 
#         'learning_rate': 0.0001, 
#         'embedding_dim': 64, 
#         'hidden_size': 64, 
#         'num_layers': 1, 
#         'num_blocks': 1, 
#         'num_heads': 1 }
# ]

hyperparam_sets = [generate_hyperparam_set() for i in range(10)]
# ============================================

for idx, hype in enumerate(hyperparam_sets): 

  filename = f"{t}-{n}-{m}/model-{idx}.log"
  _log(filename, "-------------------------------", mode="w")  # this will clean existing content, if any

  model, losses_by_epoch, elapsed_time = tuner(train_set, model_class, 
                                                **hype, 
                                                log=log, filename=filename)

  if save_model: 
    model_path = f"{t}-{n}-{m}/model-{idx}.pth"
    torch.save(model, model_path)                                                

  loss_t, acc_t, f1_macro_t = evaluate_model(model, train_set, criterion)
  loss_d, accuracy_d, f1_macro_d = evaluate_model(model, dev_set, criterion)

  eval_t = f"[train] Loss: {loss_t}, Accuracy: {acc_t}, F1-macro: {f1_macro_t}"
  eval_d = f"[dev]   Loss: {loss_d}, Accuracy: {accuracy_d}, F1-macro: {f1_macro_d}"

  print(eval_t)
  print(eval_d)

  if log: 
    _log(filename, eval_t)
    _log(filename, eval_d)
