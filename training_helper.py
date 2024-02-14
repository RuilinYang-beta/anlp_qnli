import torch
import torch.nn as nn
import string
import time
import unidecode
import matplotlib.pyplot as plt

from utils import char_tensor, random_training_set, time_since, random_chunk, CHUNK_LEN
from evaluation import compute_bpc
from model.model import LSTM

"""
I get this file from assignment 3, deleted some functions. 
Now the most relevant thing is `train` fucntion. 
Later on we might borrow something from `tuner` and `custom_train` for hyperparam tuning. 
"""


def train(decoder, decoder_optimizer, inp, target):
    """
    We need to adapt this function, things to consider: 
    -> how is the loss computed? Is it computed from the output of the last token? Or is it computed from the pooled output of each token?
       More on this see textbook page 193 near the end of the page.
    """
    # hidden, cell = decoder.init_hidden()
    # decoder.zero_grad()
    # loss = 0
    # criterion = nn.CrossEntropyLoss()

    # for c in range(CHUNK_LEN):
    #     output, (hidden, cell) = decoder(inp[c], (hidden, cell))
    #     # `output` of shape [1, 100]
    #     # `target[c].view(1)` of shape [1] --> index of target char
    #     # nn.CrossEntropyLoss() 
    #     # -> will apply softmax to `output` before computing loss
    #     # -> and will compare the output with `target[c].view(1)` (which is the index of target char)
    #     loss += criterion(output, target[c].view(1))

    # # nudge params per chunk
    # loss.backward()
    # decoder_optimizer.step()

    # return loss.item() / CHUNK_LEN


# def tuner(n_epochs=3000, print_every=100, plot_every=10, 
#           # params of model
#           hidden_size=128, 
#           n_layers=2,
#           lr=0.005, 
#           # params of generate()
#           start_string='A', 
#           prediction_length=100, 
#           temperature=0.8
#           ):
#         # YOUR CODE HERE
#         #     TODO:
#         #         1) Implement a `tuner` that wraps over the training process (i.e. part
#         #            of code that is ran with `default_train` flag) where you can
#         #            adjust the hyperparameters
#         #         2) This tuner will be used for `custom_train`, `plot_loss`, and
#         #            `diff_temp` functions, so it should also accomodate function needed by
#         #            those function (e.g. returning trained model to compute BPC and
#         #            losses for plotting purpose).

#         ################################### STUDENT SOLUTION #######################
#         all_characters = string.printable
#         n_characters = len(all_characters)

#         decoder = LSTM(n_characters, hidden_size, n_characters, n_layers)   
#         decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)

#         start = time.time()
#         all_losses = []
#         loss_avg = 0

#         for epoch in range(1, n_epochs+1):
#             loss = train(decoder, decoder_optimizer, *random_training_set())    
#             loss_avg += loss

#             if epoch % print_every == 0:
#                 print('[{} ({} {}%) {:.4f}]'.format(time_since(start), epoch, epoch/n_epochs * 100, loss))
#                 print(generate(decoder, start_string, prediction_length, temperature=temperature), '\n')            # !!!!

#             if epoch % plot_every == 0:
#                 all_losses.append(loss_avg / plot_every)
#                 loss_avg = 0

#         # perhaps return the trained model
#         return decoder, all_losses
#         ############################################################################


# def custom_train(hyperparam_list):
#     """
#     Train model with X different set of hyperparameters, where X is 
#     len(hyperparam_list).

#     Args:
#         hyperparam_list: list of dict of hyperparameter settings

#     Returns:
#         bpc_dict: dict of bpc score for each set of hyperparameters.
#     """
#     TEST_PATH = './data/dickens_test.txt'
#     string = unidecode.unidecode(open(TEST_PATH, 'r').read())   # len is 5476
#     # YOUR CODE HERE
#     #     TODO:
#     #         1) Using `tuner()` function, train X models with different
#     #         set of hyperparameters and compute their BPC scores on the test set.

#     ################################# STUDENT SOLUTION ##########################
#     bpc_dict = {}

#     for idx, hyperparam_dict in enumerate(hyperparam_list):
#         decoder, _ = tuner(**hyperparam_dict)
#         bpc_dict[f"hyperparamset-{idx}"] = compute_bpc(decoder, string)

   
#     return bpc_dict
#     #############################################################################
