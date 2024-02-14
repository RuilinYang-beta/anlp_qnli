import argparse
import torch
import torch.nn as nn
import unidecode
import string
import time

from utils import char_tensor, random_training_set, time_since, CHUNK_LEN
from language_model import plot_loss, diff_temp, custom_train, train, generate
from model.model import LSTM


def main():
    parser = argparse.ArgumentParser(
        description='Stress Test'
    )

    parser.add_argument(
        '--rnn', dest='rnn',
        help='Train a simple RNN model',
        action='store_true'
    )

    parser.add_argument(
        '--rnn_bi', dest='rnn_bi',
        help='Train a BiRNN model',
        action='store_true'
    )

    parser.add_argument(
        '--rnn_stacked', dest='rnn_stacked',
        help='Train a stacked RNN model',
        action='store_true'
    )

    args = parser.parse_args()



    """
    TODO: We still need: 
    -> a function to load the train/dev/test data, and if necessary, preprocessing 
    """

    if args.rnn:
        # -> some hyperparams 
        # -> init model and optimizer 
        # -> train model

        """
        A epoch is a full pass over the entire training set.
        A batch is a subset of the training set that is used to estimate the error of the model.
        """
        
        """
        TODO: We still need: 
        -> a function to segment the training set into batches
        -> a function to nudge the params per batch -> could be adapted from `train` function
        -> a function to evaluate the trained model against dev/test data
        
        We optionally need: 
        -> some plotting function to plot the loss
        -> some timing function 
        """         
        # for each epoch 
            # for each batch 
                # do forward thing
                # get loss 
                # do backward thing to nudge params 

        # evaluate the trained model against dev/test data

        # TODO: below won't work, just for reference  
        # for epoch in range(1, n_epochs+1):
        #     # this function nudges params per batch size 
        #     loss = train(decoder, decoder_optimizer, *random_training_set())    
        #     # accumulate loss per `plot_every` param updates
        #     loss_total += loss    # -> for plotting/seeing the loss is actually decreasing
        pass

    if args.rnn_bi:
        # more or less the same as above 
        pass

    if args.rnn_stacked:
        # more or less the same as above 
        pass

    """
    We perhaps also need this flag for trying hyper params. 
    Can we make educated guess which combinations of params to try? How? 
    """
    # if args.custom_train:
    #     # YOUR CODE HERE
    #     #     TODO:
    #     #         1) Fill in `hyperparam_list` with dictionary of hyperparameters
    #     #         that you want to try.
    #     ####################### STUDENT SOLUTION ###############################
    #     hyperparam_list = [{"n_epochs":100, 
    #                         "print_every":10, 
    #                         "hidden_size": 10}, 
    #                        {"n_epochs":100, 
    #                         "print_every":10, 
    #                         "hidden_size": 20}]
    #     ########################################################################
    #     bpc = custom_train(hyperparam_list)

    #     for keys, values in bpc.items():
    #         print("BPC {}: {}".format(keys, values))



if __name__ == "__main__":
    main()
