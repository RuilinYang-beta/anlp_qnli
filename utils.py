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

def load_data(dataset: str): 
  with open(f"{BASE_PATH}{dataset}.json", 'r') as file:
      data_dict = json.load(file)

  # Access the data
  return data_dict

def pretty_print(data: dict):
  print(json.dumps(data, indent=4))


def build_vocabulary(dataset, notation: Notation, min_freq=None):
  """
  Get vocabulary based on dataset and notation, where notation is one of the Notation enum values.
  `min_freq` is the minimum frequency of a token to be included in the vocabulary, if it's None, then all tokens are included.
  setting `min_freq` is supposed to add model generalizability. 
  """
  if notation in notation2key:
    keys = notation2key[notation]
  else: 
    throw("Notation not supported")

  words_raw = [token.lower() for record in dataset 
                              for key in keys
                              for token in record[key].split(" ")]

  words_count = Counter(words_raw)                            

  if min_freq:
    vocab = list(set(token if count > min_freq else Token.Unk for token, count in words_count.items()))
  else: 
    vocab = list(words_count.keys())
    vocab.append(Token.UNK)

  vocab.append(Token.SEP)

  return vocab


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

    for i, token in enumerate(s1_tokens):
      x_tensor[i] = vocab.index(token)

    x_tensor[len(s1_tokens)] = vocab.index(Token.SEP)

    for i, token in enumerate(s2_tokens):
      x_tensor[i + len(s1_tokens) + 1] = vocab.index(token)

    y_tensor = torch.tensor(labels.index(record["answer"]))

    yield x_tensor, y_tensor