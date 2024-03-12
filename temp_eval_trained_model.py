import torch
import torch.nn as nn

from statics import Notation, Token, notation2key, labels
from utils import load_data, pretty_print, build_vocabulary, xy_pair_generator  
from evaluation import evaluate_model
from statics import DEVICE, SEED

"""
Here load the trained model and evaluate. 
It's also possible to load the model and continue training: 

# To load the model back later, you can use:
# loaded_model = torch.load(model_path)

# To continue training, you can create a new instance of optimizer,
# and point it to the parameters of the loaded model, like this:
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
... and continue the training loop
"""

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


model_path = 'trained_models/simple_rnn_b_best_noPack.pth'
loaded_model = torch.load(model_path).to(DEVICE)  
print(f"~~~~~~~ model: {model_path} loaded ~~~~~~~")


# some constants, copied by hand from `temp_simplernn_playground.py`
# they have to be the same with the ones used for training
notation = Notation.ORIGINAL_CHAR
hidden_size = 50
criterion = nn.CrossEntropyLoss()

# get vocab based on dataset and notation -> change to StressDataset
train = load_data("train")
vocab = build_vocabulary(train, notation)

# evaluate on training set -> supposedly performs better than on dev
eval_loader = xy_pair_generator(train, notation, vocab)   # TODO: migrate to dataset
loss, accuracy, f1_macro = evaluate_model(loaded_model, 
                                          loaded_model.init_hidden((1, hidden_size)).to(DEVICE), 
                                          eval_loader,  # -> change to train_data_set
                                          criterion)
print(f"[train] Loss: {loss}, Accuracy: {accuracy}, F1-macro: {f1_macro}")

# evaluate on dev set
dev = load_data("dev")
eval_loader = xy_pair_generator(dev, notation, vocab)  # TODO: migrate to dataset
loss, accuracy, f1_macro = evaluate_model(loaded_model, 
                                          loaded_model.init_hidden((1, hidden_size)).to(DEVICE), 
                                          eval_loader, criterion)
print(f"[dev]   Loss: {loss}, Accuracy: {accuracy}, F1-macro: {f1_macro}")
