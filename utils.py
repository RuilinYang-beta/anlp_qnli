"""
Utility functions to : 
- load data from file
- build vocabulary based on dataset and notation
- yield a pair of (input, label) for each record in the dataset to feed to model
"""

import json 
from collections import Counter
import torch 

from statics import Notation, Token, notation2key, labels

BASE_PATH = "data/"


# [done] moved to StressDataset
def load_data(dataset: str): 
  with open(f"{BASE_PATH}{dataset}.json", 'r', encoding='utf-8') as file:
      data_dict = json.load(file)

  # Access the data
  return data_dict

def pretty_print(data: dict):
  print(json.dumps(data, indent=4))

# [done] moved to StressDataset, with some modification
def build_vocabulary(dataset, notation: Notation, min_freq=None):
  """
  Get vocabulary based on dataset and notation, where notation is one of the Notation enum values.
  `min_freq` is the minimum frequency of a token to be included in the vocabulary, if it's None, then all tokens are included.
  setting `min_freq` is supposed to add model generalizability. 
  """
  if notation in notation2key:
    keys = notation2key[notation]
    # avoid modifying the keys in place
    keys = [k for k in keys] 
    keys.append("options") 
  else: 
    raise ValueError("Notation not supported")

  words_raw = [token.lower() for record in dataset 
                              for key in keys
                              for token in record[key].split(" ")]

  # high frequency words first to save lookup time
  words_count = Counter(words_raw)                            
  words_count = dict(sorted(words_count.items(), key=lambda item: item[1], reverse=True))

  if min_freq:
    vocab = list(set(token for token, count in words_count.items() if count > min_freq))
  else: 
    vocab = list(words_count.keys())

  # add special tokens to the front since they probably have high frequency 
  vocab.insert(0, Token.UNK)
  vocab.insert(0, Token.SEP)
  # !! padding token MUST be at index 0, 
  # since `pad_sequence` will pad a batch of sequence with 0, 
  # we want token 0 not to tamper with real tokens
  vocab.insert(0, Token.PADDING)  

  return vocab

# [done] separated generating xy_pair function, and data preprocessing
# generating xy_pair moved to Dataloader, 
# data preprocessing moved to data_utils/transforms.py
def xy_pair_generator(dataset, notation, vocab): # -> dataloader thing??? batch thing???
  """
  Yield a pair of (input, label) for each record in the dataset, which is ready to be fed to the model.
  `input` is a tensor of token indices, for tokens in [statement1_tokens, separate_token, statement2_tokens]
  `label` is a tensor of label index
  """
  keys = notation2key[notation]

  for record in dataset: 
    key1, key2 = keys
    s1_tokens = record[key1].lower().split(" ")
    s2_tokens = record[key2].lower().split(" ")
    x_tensor = torch.zeros(len(s1_tokens) + len(s2_tokens) + 1).long()   # concat 2 statements and a separator token
    # ------ with option ------
    # option_tokens = record["options"].lower().split(" ")
    # x_tensor = torch.zeros(len(s1_tokens) + len(s2_tokens) 
    #                        + len(option_tokens) + 2).long()   # +2 is for 2 separator tokens

    # token in s1 and the separation token that follows
    for i, token in enumerate(s1_tokens):
      try: 
        x_tensor[i] = vocab.index(token)
      except ValueError: 
        x_tensor[i] = vocab.index(Token.UNK)
    x_tensor[len(s1_tokens)] = vocab.index(Token.SEP)

    # token in s2
    for i, token in enumerate(s2_tokens):
      try: 
        x_tensor[i + len(s1_tokens) + 1] = vocab.index(token)
      except ValueError: 
        x_tensor[i + len(s1_tokens) + 1] = vocab.index(Token.UNK)


    # ------ with option ------
    # # s1 and the separation token that follows
    # for i, token in enumerate(s1_tokens):
    #   x_tensor[i] = vocab.index(token)
    # x_tensor[len(s1_tokens)] = vocab.index(Token.SEP)

    # # s2 and the separation token that follows
    # for i, token in enumerate(s2_tokens):
    #   x_tensor[i + len(s1_tokens) + 1] = vocab.index(token)
    # x_tensor[len(s1_tokens) + len(s2_tokens) + 1] = vocab.index(Token.SEP)

    # # options
    # for i, token in enumerate(option_tokens):
    #   x_tensor[i + len(s1_tokens) + len(s2_tokens) + 2] = vocab.index(token)


    y_tensor = torch.tensor(labels.index(record["answer"]))

    yield x_tensor, y_tensor