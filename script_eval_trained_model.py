import torch
import torch.nn as nn

from data_loading.StressDataset import StressDataset
from data_loading.transforms import transform1, target_transform

from statics import DEVICE, SEED, Notation
from evaluation import evaluate_model

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


model_path = 'train-ori_char-FeedForwardNN/model-0.pth'
model = torch.load(model_path).to(DEVICE)  
print(f"~~~~~~~ model: {model_path} loaded ~~~~~~~")

# === prepare for training and tuning ===
TRAIN_PATH = 'data/train.json'
NOTATION = Notation.ORIGINAL_CHAR

train_set = StressDataset(TRAIN_PATH, NOTATION, forEval=False,
                        transform=transform1, 
                        target_transform=target_transform)

dev_set = StressDataset('data/dev.json', NOTATION, forEval=True,
                        transform=transform1, 
                        target_transform=target_transform, 
                        vocab=train_set.get_vocab())

criterion = nn.CrossEntropyLoss(reduction='sum')

loss_t, acc_t, f1_macro_t = evaluate_model(model, train_set, criterion)
loss_d, accuracy_d, f1_macro_d = evaluate_model(model, dev_set, criterion)

eval_t = f"[train] Loss: {loss_t}, Accuracy: {acc_t}, F1-macro: {f1_macro_t}"
eval_d = f"[dev]   Loss: {loss_d}, Accuracy: {accuracy_d}, F1-macro: {f1_macro_d}"

print(eval_t)
print(eval_d)
