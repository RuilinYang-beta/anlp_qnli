from .FeedForwardNN import FeedForwardNN
from .SimpleRNN import SimpleRNN
from .BiRNN import BiRNN
from .SimpleTransformer import SimpleTransformer  

import sys
sys.path.append('..')  # Add the parent directory to the sys.path

from statics import DEVICE

"""
Centralize model initialization here.
Args: 
- All possible hyperparams for all models.
Return:
- The model and the hyperparams that are specific to the model.
"""

class ModelFactory:

  @staticmethod
  def init_model(model_class, 
            vocab_size=None, 
            # --- common hyperparams - for model shape ---
            embedding_dim=None,
            # --- [model specific hyperparams] ---
            hidden_size=None,      # for RNN and FFNN
            num_layers=None,         # for RNN and FFNN  
            num_blocks=None,         # for Transformer
            num_heads=None,          # for Transformer
            # -------------------------------------
            output_size=3,
            dropout=0.2
            ):

    if model_class.__name__ == "SimpleRNN" or model_class.__name__ == "BiRNN":
      return (model_class(vocab_size, embedding_dim, hidden_size, output_size, 
                          dropout=dropout,
                          num_layers=num_layers).to(DEVICE), 
              {
                "hidden_size": hidden_size, 
                "num_layers":  num_layers
              }
            )


    elif model_class.__name__ == "FeedForwardNN": 
      return (model_class(vocab_size, embedding_dim, hidden_size, output_size, 
                    dropout=dropout, num_layers=num_layers).to(DEVICE), 
              {
                "hidden_size": hidden_size, 
                "num_layers":  num_layers
              }
              )

    elif model_class.__name__ == "SimpleTransformer": 
      return (model_class(vocab_size, embedding_dim,
                    num_blocks, num_heads,
                    output_size, 
                    dropout=dropout).to(DEVICE),
              {
                "num_blocks": num_blocks, 
                "num_heads": num_heads
              }
      )

    else: 
      raise TypeError(f"Model type not supported")


