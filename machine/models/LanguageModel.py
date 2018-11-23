
import torch.nn as nn
import torch.nn.functional as F

from .baseModel import BaseModel

class LanguageModel(BaseModel):
    """
    Implements a language model
    """
    
    def flatten_parameters(self):
        raise NotImplementedError("Function should be implemented")

    def forward(self, inputs, input_lengths=None):
        raise NotImplementedError("Language model should be implemented")
     
