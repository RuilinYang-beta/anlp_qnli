# About

In this project we explore factors that might have an impact on model performance of QNLI task. The considered factors are:

- If the traing set is augmented
- Notation of numerals
- Model architecture
- Hyperparameters
  - learning rate
  - batch size
  - embedding dimension
  - hidden size (only for FFNN, RNN, BiRNN)
  - number of layers (only for FFNN, RNN, BiRNN)
  - number of blocks (only for Transformer)
  - number of heads (only for Transformer)

We train models under different configurations, then zoom in to see if hyperparameters of the best-performed and worst-performed model shows a pattern.

All the trained models can be accessed here: https://drive.google.com/drive/folders/1JPjLZY4nAO8t9UJp8XItxl8Irth-lW9W?usp=sharing

# Dependencies

- pytorch 2.2.1

# Usage

Please use command line arguments to set the choice of training set, notation of number, and model architecture, plus whether to save the trained model, and whether to display log in the terminal.

`python qnli.py {normal, augmented} {original, character} {FFNN, RNN, BiRNN, Transformer} [-e <num_epochs>] [-n <num_sets>] [-s] [-l]`

In the {curly braces} are positional arguments and its options, in [square brackets] are optional arguments.

- {normal, augmented}: choose if use the normal training set or the augmented training set; augment training set is computed by duplicating the examples where the answer is "neutral" or "contradiction", and swap the position of s1 and s2.
- {original, character}: choose how the numerals in the sentence is tokenized. "original" for each numeral as a token, "character" for each numeral is split into multiple digit-based tokens.
- {FFNN, RNN, BiRNN, Transformer}: choose model architecture
- [-e <num_epochs>]: number of epochs
- [-n <num_sets>]: number of sets of hyperparameters to generate, one set will be used to initialize and train one model
- [-s]: whether to save the trained model to a file
- [-l]: whether to save the log of training process to a file

See also `python qnli.py -h` for help message.

Example usage:

`python qnli.py augmented character BiRNN -e 1000 -n 5 -l -s`

# File structure

## Files in the root folder

- `qnli.py`: Entry point of the program.
- `training_helper.py`: scaffolding code for training loop.
- `statics.py`, `utils.py`, `evaluation.py`: they are self-descriptive
- `script_augment_train_data.py`: the script we use to augment training set
- `script_eval_trained_model.py`: load and evaluate a trained model
- `plot.ipynb`: the plotting jupyter notebook, note it assumes specific paths to model, see the instructions in the notebook

## `data/`

Contains data.

## `data_analysis/`

Contains another copy of data, and an exploratory data analysis `data_analysis/very_rough_data_analysis.ipynb`

## `data_loading/`

Our custom `Dataset` class, preprocessing functions, and batch collate functions.

## `models/`

Contains model classes: FFNN, RNN, BiRNN, Transformer. And other utility classes such as ModelFactory and CustomSequential.

## `trained_models/`

Contain trained models, and for each model a .md file recording their decrease of loss.
