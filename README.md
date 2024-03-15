# What's new

## `training_helper.py` for training and hyperparam tuning

In this file,

- `train` is the training process for an epoch,
- `tuner` is a bunch of hyperparams, epoch loops, inside the epoch loop it calls `train`
  The purpose of `tuner` is to encapsulate the hyperparams, and decouple `train` from hyperparams as much as possible.

In some situation it's inevitable, eg. in `train`, the `forward` method of SimpleRNN requires `hidden_size` hyperparam. To avoid clutter, these model-specific hyperparams are caputured in `**kwargs`, so that the positional arguments / named arguments of `train` are the ones that are common across all models.

There is detailed TODOs in the comments of this file, they are assigned to either Ellie or Ruilin.

## train and evaluate model in `temp.py`

- `python temp.py`

It's making use of `training_helper.py`

## train and evaluate model in `temp_simplernn_playground.py`

- `python temp_simplernn_playground.py` to train a simple rnn, and save the trained model

It's robust to play around hyperparams now, but slow, not recommended. Will be deprecated After BiRNN is safely integrated.

## load and evaluate trained model `python temp_eval_trained_model.py`

It's self-descriptive.

# what we have now

## the `temp_`s in root folder

All files starts with "temp\_" are for dev purposes, they will be cleaned up gradually and won't be a part of the final delivery

- `temp.py` to play with SimpleRNN with Dataset and Dataloader, and save the trained model
- `temp_simplernn_playground.py` to play with SimpleRNN in old way (see above "What's new" section), and save the trained model
- `temp_eval_trained_model.py` to load and evaluate a trained model, it's also possible to continue training the model

- `temp_birnn_playground.py` to play with BiRNN, something might be broken, we will fix it later
- `temp_evals.py` is a demo of evaluation function
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

## Other files in root directory

- `evaluation.py`: contains the evaluation function.
- `qnli.py`: copied the structure from assignments, maybe good for final delivery, who knows?
- `script_augment_train_data.py`: augment training data by swapping s1 s2 (for all 4 notations), -> perhaps better move to some .ipynb file
- `statics.py`: containing statics, will be deprecated when we fully migrate to Dataloader way, now it's still useful for two temp_playground file
- `training_helper.py`: likely will contain a function for training loop, a function for evaluating, a function for hyperparam tuning, a function for plotting, a function for timing, etc..
- `utils.py` will be deprecated. More info see the file.

# what to do next?

- [high] Hyperparam tuning
- [high] Integrate BiRNN / StackRNN to simpleRNN
- [high] clean code: training always batch, should eval be unbatched or batched?
- [high] unified Evaluation function for all models
- [mid] NEKG embeddings
- [mid] error analysis -> can play with `simple_rnn_b_best_noPack.pth` but don't be too serious
- [mid] plotting -> make paper look nice
- [low] add type signature to functions for easy maintenance
- [low] something about speed, recorded in trello
