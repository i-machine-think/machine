import torch

class AttentionTargets(object):
    """
    Base class for encapsulation of functions generating attentive guidance.

    This class defines interfaces that should be implemented to interact with
    the AttentionTrainer class.

    Note:
        Do not use this class directly, use one of the sub classes.

    Args:

    Attributes:
        name (str): name of the attention generation mechanism
        key (str): key under which targets are stored in target dict
        pad_token (int): target token to be ignored

    """

    def __init__(self, name, key, pad_token=-1):
        self.name = name
        self.key = key
        self.pad_token = pad_token

    def add_attention_targets(self, input_variables, input_lengths, target_variables):
        """ Generate attention targets
        
        This  method defines how to compute the attention targets given the input
        variables, and adds it to the inputted dictionary containing previously
        defined target variables.

        Args:
            input_variables (torch.Tensor): inputs to a batch
            input_lengths (torch.Tensor): input lengths
            target_variables (dict): mapping keys to torch.Tensors representing target variables

        Returns:
            target_variables (dict): dictionary with attention targets added

        """
        raise NotImplementedError("Implement in subclass")


class LookupTableAttention(AttentionTargets):
    """ Attention for lookup tables postfix annotation

    """
    _NAME = "lookup_table"
    _KEY = "attention_target"

    def __init__(self):
        super(LookupTableAttention, self).__init__(name=self._NAME, key=self._KEY, pad_token=-1)

    def add_attention_targets(self, input_variables, input_lengths, target_variables):
        # update target variables with attention targets
        target_variables['attention_target'] = torch.zeros_like(target_variables['decoder_output'])
        return target_variables
