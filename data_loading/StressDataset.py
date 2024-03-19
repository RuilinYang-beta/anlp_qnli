import json
from collections import Counter
from torch.utils.data import Dataset

import sys
sys.path.append('..')  # Add the parent directory to the sys.path
from statics import Notation, notation2key, Token, labels


"""
Note: 
before `utils.py` is completely deprecated, 
keep `load_data` and `build_vocabulary` there in sync.
"""
class StressDataset(Dataset):

  def __init__(self, filepath, notation, forEval: bool,
              transform=None, target_transform=None,
              vocab=None, min_freq=None   # params for building vocabulary
              ):
    """
    Args: 
    - filepath: path to the json file
    - notation: one of the Notation enum
    - forEval: whether the dataset is for evaluation or not
              if False, `vocab` should be None; if True, `vocab` should be provided
    - transform: transform to apply to the data
    - target_transform: transform to apply to the target
    - vocab: vocabulary to use for the dataset, 
            it should be None for training set, since the vocab is built from the training set;
            for dev and test set, the vocab built from corresponding training set should be provided.
    - min_freq: minimum frequency of a token to be included in the vocabulary, 
                if it's None, then all tokens are included. Only used when `vocab` is None.
    """

    # stop early if error
    if notation not in notation2key:  
      raise ValueError("Notation not supported.")   

    if forEval is None: 
      raise ValueError("Please specify whether the dataset is for evaluation or not by setting `forEval` to True or False.")

    if forEval and vocab is None:
      raise ValueError("Vocabulary computed from training set must be provided for evaluation dataset.")

    if not forEval and vocab is not None:
      raise ValueError("Vocabulary should be None for building training set.")

    with open(filepath, 'r',  encoding='utf-8') as json_file:
      self.data_dicts = json.load(json_file)

    self.notation = notation
    self.transform = transform
    self.target_transform = target_transform

    if vocab:
      self.vocab = vocab
    else:
      self.vocab = self.build_vocabulary(self.data_dicts, notation, min_freq)

    self.vocab_size = len(self.vocab)


  def __len__(self):
    return len(self.data_dicts)


  def __getitem__(self, idx):
      record = self.data_dicts[idx]
      x_tensor = None
      y_tensor = None


      if self.transform:
          # give `transform` as many as possible things, let it decide what to use
          x_tensor = self.transform(record, self.notation, self.vocab) 
      if self.target_transform:
          y_tensor = self.target_transform(record)
      
      return x_tensor, y_tensor


  def build_vocabulary(self, data_dicts, notation, min_freq):
    """
    Args: 
    - data_dicts: list of records, where each record is a dictionary
    - notation: one of the Notation enum
    - min_freq: minimum frequency of a token to be included in the vocabulary, 
                if it's None, then all tokens are included.
                setting `min_freq` is supposed to add model generalizability. 
    """

    keys = notation2key[notation]
    
    # some questions only have two possible answers, 
    # eg. records from AWPNLI can only be entailment / contradiction, 
    #     records from NewsNLI / RTE_Quant can only be entailment / neutral
    # we may want to add this field to x_tensor, so it needs to be in vocabulary
    # and let `transform` function decide whether to use it
    keys = [k for k in keys]
    keys.append("options") 

    words_raw = [token.lower() for record in data_dicts 
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

    
  def get_vocab(self):
    return self.vocab


  def get_vocab_size(self):
    return self.vocab_size