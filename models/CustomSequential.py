import torch.nn as nn

class CustomSequential(nn.Sequential): 
  """ 
  A custom nn.Sequential that can pass more than one param to its modules. 
  """
  
  def forward(self, *input):
    for module in self._modules.values():
      output = module(*input)
    return output
