import time
import json 
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader

from data_loading.StressDataset import StressDataset
from data_loading.transforms import transform1, target_transform

from training_helper import tuner, generate_hyperparam_set
from utils import prepare_parser, _log
from models.ModelFactory import ModelFactory
from models.FeedForwardNN import FeedForwardNN
from models.SimpleRNN import SimpleRNN
from models.BiRNN import BiRNN
from models.SimpleTransformer import SimpleTransformer

from evaluation import evaluate_model
from statics import DEVICE, SEED, Notation

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# === get choice of training_set, notation, model, save, log from command line ===
parser = prepare_parser()
args = parser.parse_args()

# === init configs ===
model_map = {"FFNN": FeedForwardNN, "RNN": SimpleRNN, "BiRNN": BiRNN, "Transformer": SimpleTransformer}
# --- for data --- 
TRAIN_PATH = 'data/train.json' if args.training == 'normal' else 'data/train_augmented.json'
NOTATION = Notation.ORIGINAL if args.notation == 'original' else Notation.ORIGINAL_CHAR
# --- for model ---
model_class = model_map[args.model]
# --- for log / save model ---
save_model = True if args.save else False
log = True if args.log else False

# === prepare for training and tuning ===
train_set = StressDataset(TRAIN_PATH, NOTATION, forEval=False,
                        transform=transform1, 
                        target_transform=target_transform)

dev_set = StressDataset('data/dev.json', NOTATION, forEval=True,
                        transform=transform1, 
                        target_transform=target_transform, 
                        vocab=train_set.get_vocab())

criterion = nn.CrossEntropyLoss(reduction='sum')

# === prepare for log/save model  ===
t = "train" if TRAIN_PATH == 'data/train.json' else "train_aug"
n = NOTATION.value
m = model_class.__name__

# === set hyperparams ===
# --- supported hyperparams ---
# * batch_size
# * learning_rate
# * embedding_dim
# * hidden_size       only FFNN / RNN
# * num_layers        only FFNN / RNN
# * num_blocks        only Transformer 
# * num_heads         only Transformer

# --- option1: hand-picked hyperparams, use it for test/zoom-in ---
# hyperparam_sets = [ 
#   # { 
#   # 'batch_size': 300, 
#   # 'learning_rate': 0.0001, 
#   # 'embedding_dim': 64, 
#   # 'hidden_size': 64, 
#   # 'num_layers': 1, 
#   # 'num_blocks': 1, 
#   # 'num_heads': 1 
#   # }, 
#   # { 
#   # 'batch_size': 500, 
#   # 'learning_rate': 0.01726287996407694, 
#   # 'embedding_dim': 128, 
#   # 'hidden_size': 192, 
#   # 'num_layers': 3, 
#   # 'num_blocks': 1, 
#   # 'num_heads': 1 
#   # } 
# ]

# --- option2: generate random hyperparams ---
hyperparam_sets = [generate_hyperparam_set() for i in range(args.num_sets)]

# === train models with diff hyperparam sets ===
for idx, hype in enumerate(hyperparam_sets): 

  filename = f"{t}-{n}-{m}/model-{idx}.log"

  if log: 
    _log(filename, "===============================", mode="w")  # this will clean existing content, if any
    _log(filename, "Training with the following config: ")
    _log(filename, json.dumps(dict(vars(args)), indent=4))
    _log(filename, "-------------------------------")  
  
  print("===============================")
  print("Training with the following config: ")
  print(json.dumps(dict(vars(args)), indent=4))
  print("-------------------------------")

  model, losses_by_epoch, elapsed_time = tuner(train_set, model_class, 
                                                n_epochs=args.epochs,
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
