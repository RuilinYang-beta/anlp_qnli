from .FeedForwardNN import FeedForwardNN
from .SimpleRNN import SimpleRNN
from .SimpleTransformer import SimpleTransformer  

import sys
sys.path.append('..')  # Add the parent directory to the sys.path

from statics import DEVICE


class ModelFactory:

  @staticmethod
  def init_model(model_class, 
            vocab_size=None, 
          # --- common hyperparams - for model shape ---
          embedding_dim=None,
          output_size=3,      # let's fix it to be 3, we probably don't have time to experiment 2-binary classification
          dropout=None,
          # --- [model specific hyperparams] ---
          hidden_size=None,      # for RNN and FFNN
          num_layers=None,         # for RNN and FFNN  (TODO: enable FFNN to have stacked layers)
          bidirectional=None,  # for RNN
          num_blocks=None,         # for Transformer
          num_heads=None,          # for Transformer
            ):

    m = model_class.get_type()

    if m == "RNN":
      return model_class(vocab_size, embedding_dim, hidden_size, output_size, 
                    dropout=dropout,
                    num_layers=num_layers, bidirectional=bidirectional).to(DEVICE)

    elif m == "FFNN": 
      return model_class(vocab_size, embedding_dim, hidden_size, output_size, 
              dropout=dropout, num_layers=num_layers).to(DEVICE)

    elif m == "TRANSFORMER": 
      return model_class(vocab_size, embedding_dim,
                    num_blocks, num_heads,
                    output_size, 
                    dropout=dropout).to(DEVICE)

    else: 
      raise TypeError(f"Model type not supported")
