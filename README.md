# What's new

## the omnipotent `temp.py`

- `python temp.py`

It's making use of `training_helper.py`. FFNN, RNN, Transformer they all work fine now.
All supported hyperparams are listed in the comments of this file.

## RNN family

- We can have BiRNN and StackedRNN by passing flags to `SimpleRNN` class, see `temp.py` and `models/SimpleRNN.py`.
- RNN don't need h_0 to call `forward` now, this leads to less cluttered code -> PyTorch will handle it

## `training_helper.py` for training and hyperparam tuning

In this file,

- `train` is the training process for an epoch,
- `tuner` is a bunch of hyperparams, epoch loops, inside the epoch loop it calls `train`
  The purpose of `tuner` is to encapsulate the hyperparams, and decouple `train` from hyperparams as much as possible.

There is detailed TODOs in the comments of this file, they are assigned to either Ellie or Ruilin.

# what we have now

## the `temp_`s in root folder

All files starts with "temp\_" are for dev purposes, they will be cleaned up gradually and won't be a part of the final delivery

- `temp.py` to play with all models and all supported hyperparameters, optionally save the model, and evaluate model
- `temp_pytorch.py` to play with your GPU

## `models/`

Contains model classes.

## `trained_models/`

Contain trained models, and for each model a .md file recording their decrease of loss.

## `temp_sound/`

A lovely sound to play when model finishes training.
Need to install `playsound` package, by `pip install playsound`.

## `data_loading/`

Our custom Dataset class, preprocessing functions, and static values.

## `data_analysis/`

Data analysis things.

## `data`

Contains data.

## scripts

One-time scripts of some data wrangling, you don't need to care about it.

## Other files in root directory

- `training_helper.py`: contain the `train` function for training 1 epoch, a `tuner` function calling train with diff hyperparams; later on likely will contain more things for hyperparam tuning and plotting
- `evaluation.py`: contains the evaluation function.
- `statics.py`: containing statics
- `qnli.py`: our final entry point of the code
- `utils.py`: functions there have all migrated, but please keep it here for now in case some scripts need it.

# what to do next?

- [high] Hyperparam tuning -> it's a big time consuming thing
- [mid] NEKG embeddings
- [mid] error analysis -> can play with `simple_rnn_b_best_noPack.pth` but don't be too serious
- [mid] plotting -> make paper look nice
- [low] add type signature to functions for easy maintenance
- [low] something about speed, recorded in trello
