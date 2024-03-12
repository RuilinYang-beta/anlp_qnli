"""
Contains series of functions that transform a record into x_tensor and y_tensor
Functions starts with `transform` operate on the input record and return x_tensor;
Functions starts with `target_transform` operate on the input record and return y_tensor.

Params: 
  - record: a record from the dataset, like so
            {
              "statement1": "Patrick has a locker with a less than 8 number combination",
              "statement2": "Patrick has a locker with a 3 number combination",
              "options": " Entailment or contradiction or neutral?",
              "answer": "neutral",
              "type": "Type_7",
              "statement1_sci_10E": "Patrick has a locker with a less than 8.0000000000E+00 number combination",
              "statement1_char": "Patrick has a locker with a less than 8 number combination",
              "statement1_sci_10E_char": "Patrick has a locker with a less than 8 . 0 0 0 0 0 0 0 0 0 0 E + 0 0 number combination",
              "statement2_sci_10E": "Patrick has a locker with a 3.0000000000E+00 number combination",
              "statement2_char": "Patrick has a locker with a 3 number combination",
              "statement2_sci_10E_char": "Patrick has a locker with a 3 . 0 0 0 0 0 0 0 0 0 0 E + 0 0 number combination",
              "statement1_mask": "Patrick has a locker with a less than [Num] number combination",
              "statement2_mask": "Patrick has a locker with a [Num] number combination",
              "EQUATE": "StressTest"
            }
            for more details about each field, see `data_analysis/very_rough_data_analysis.ipynb`
  - notation: one of the Notation enum
  - vocab: vocabulary to use for the dataset (only for x_tensor)

Returns: 
  - for `transform`: x_tensor
  - for `target_transform`: y_tensor
"""


"""
Note: 
before `utils.py` is completely deprecated, 
keep `xy_pair_generator` there in sync with `transform1`.
"""

import torch 

import sys
sys.path.append('..')  # Add the parent directory to the sys.path
from statics import Notation, notation2key, Token, labels

# ==================== for transform ====================

# make record into x_tensor involves two things: 
# 1. get the necessary bits of info, like s1, s2, options,
# 2. convert them into tensor in varied ways (concat, only include diff, with option, without option, etc.)

# TODO: change name to something more meaningful
def transform1(record, notation, vocab):  
    """
    Compose x_tensor from s1, s2.
    """
    keys = notation2key[notation]

    key1, key2 = keys
    s1_tokens = record[key1].lower().split(" ")
    s2_tokens = record[key2].lower().split(" ")
    x_tensor = torch.zeros(len(s1_tokens) + len(s2_tokens) +1).long()  # concat 2 statements and a separator token

    # token in s1 and the separation token that follows
    for i, token in enumerate(s1_tokens):
      x_tensor[i] = _find_index(vocab, token)
    x_tensor[len(s1_tokens)] = vocab.index(Token.SEP)

    # token in s2
    for i, token in enumerate(s2_tokens):
      x_tensor[i + len(s1_tokens) + 1] = _find_index(vocab, token)

    return x_tensor


def transform2(record, notation, vocab):  
    """
    Compose x_tensor from s1, s2 and option.
    """
    keys = notation2key[notation]

    key1, key2 = keys
    s1_tokens = record[key1].lower().split(" ")
    s2_tokens = record[key2].lower().split(" ")
    option_tokens = record["options"].lower().split(" ")
    x_tensor = torch.zeros(len(s1_tokens) + len(s2_tokens) 
                          + len(option_tokens) + 2).long()   # +2 is for 2 separator tokens

    # s1 and the separation token that follows
    for i, token in enumerate(s1_tokens):
      x_tensor[i] = _find_index(vocab, token)
    x_tensor[len(s1_tokens)] = vocab.index(Token.SEP)

    # s2 and the separation token that follows
    for i, token in enumerate(s2_tokens):
      x_tensor[i + len(s1_tokens) + 1] = _find_index(vocab, token)
    x_tensor[len(s1_tokens) + len(s2_tokens) + 1] = vocab.index(Token.SEP)

    # options
    for i, token in enumerate(option_tokens):
      x_tensor[i + len(s1_tokens) + len(s2_tokens) + 2] = _find_index(vocab, token)

    return x_tensor



def _find_index(vocab, token):
  """
  In dev/test set, it's possible to encounter tokens that are not in the training set.
  In this case, we use the index of UNK token.
"""
  try: 
    return vocab.index(token)
  except ValueError: 
    return vocab.index(Token.UNK)


# ==================== for target_transform ====================
# probably no variation in this function, but I'll keep it here for consistency

def target_transform(record):
    """
    Compose y_tensor from answer.
    """
    return torch.tensor(labels.index(record["answer"]))