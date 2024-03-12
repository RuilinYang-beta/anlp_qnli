# What's new

## train and evaluate model in `temp.py`

- `python temp.py`

It's robust to play around hyperparams now.

You have the choice to train with "batch loop w. dataloader" or "batch loop w. dataset". The latter will be deprecated in the training process, because it does not utilize parallel computing, please comment and uncomment the correct block from line 73-145.

Tested only on SimpleRNN, BiRNN might need some dimension fix to make the model agnostic to whether or not batch training but it's very doable.

The final delivery will look more like this, with functions such as build_vocabulary coupled with Dataset, preprocessing functions (now in `data_loading/transforms.py` and `data_loading/utils.py` decoupled from Dataset and Dataloader), and shuffle function built-in to Dataloader.

The final delivery will deal with 3D tensors as input (eg. (N, L, embedding_dim) where N is batch_size, it can be 1 or anything; L is sequence length, ie. how many tokens per example), even for batch_size=1 this input is still 3D. For illustration purpose the old way to train 1 example at a time is still included, this way the input to built-in RNN is 2D as (L, embedding_dim).

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

- models

  - RNN family
    - [high] Integrate BiRNN / StackRNN to simpleRNN
  - [high] Feedforward NN
  - [mid-high] transformer thing, some learning resources see below

- Transformer thing
  - [PyTorch transformer module](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html)
  - [How ppl use PyTorch transformer module](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
  - [built-in transformer models by Huggingface](https://pytorch.org/hub/huggingface_pytorch-transformers/)
  - [Open Source Models with Hugging Face](https://learn.deeplearning.ai/courses/open-source-models-hugging-face/lesson/1/introduction)
  - coursera? Youtube?
- general

  - [high] hyperparam tuning
  - [high] error analysis -> can play with `simple_rnn_b_best_noPack.pth` but don't be too serious
  - [mid] plotting -> make paper look nice
  - [mid-low] diff way of embedding
  - [low] add type signature to functions for easy maintenance
