import torch.nn as nn

class CustomSequential(nn.Sequential): 
  """ 
  A custom nn.Sequential that can handle variable-length sequences. 
  """
  
  def forward(self, *input):
    for module in self._modules.values():
      output = module(*input)
    return output
