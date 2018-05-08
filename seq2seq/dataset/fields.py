import logging

import torchtext

class SourceField(torchtext.data.Field):
    """Wrapper class of torchtext.data.Field that forces batch_first and include_lengths to be True. 
    
    Attributes:
        eos_id: index of the end of sentence symbol.
    """

    SYM_EOS = '<eos>'

    def __init__(self, use_input_eos=False, **kwargs):
        """Initialize the datafield, but force batch_first and include_lengths to be True, which is required for correct functionality of pytorch-seq2seq.
        Also allow to include SOS and EOS symbols for the source sequence.
        
        Args:
            use_input_eos (bool, optional): Whether to append the source sequence with an EOS symbol (default: False)
            **kwargs: Description
        """
        logger = logging.getLogger(__name__)

        if kwargs.get('batch_first') is False:
            logger.warning("Option batch_first has to be set to use pytorch-seq2seq.  Changed to True.")
        kwargs['batch_first'] = True

        if kwargs.get('include_lengths') is False:
            logger.warning("Option include_lengths has to be set to use pytorch-seq2seq.  Changed to True.")
        kwargs['include_lengths'] = True

        if use_input_eos:
            kwargs['eos_token'] = self.SYM_EOS
        else:
            kwargs['eos_token'] = None

        self.eos_id = None
        super(SourceField, self).__init__(**kwargs)

    def build_vocab(self, *args, **kwargs):
        super(SourceField, self).build_vocab(*args, **kwargs)
        self.eos_id = self.vocab.stoi[self.SYM_EOS]

class TargetField(torchtext.data.Field):
    """ Wrapper class of torchtext.data.Field that forces batch_first to be True and prepend <sos> and append <eos> to sequences in preprocessing step.

    Attributes:
        sos_id: index of the start of sentence symbol
        eos_id: index of the end of sentence symbol
    """

    SYM_SOS = '<sos>'
    SYM_EOS = '<eos>'
    include_output_eos = True

    def __init__(self, include_output_eos=True, **kwargs):
        logger = logging.getLogger(__name__)
        self.include_output_eos = include_output_eos

        if kwargs.get('batch_first') == False:
            logger.warning("Option batch_first has to be set to use pytorch-seq2seq.  Changed to True.")
        kwargs['batch_first'] = True

        if kwargs.get('preprocessing') is None:
            func = lambda seq: seq
        else:
            func = kwargs['preprocessing']

        if self.include_output_eos:
            kwargs['preprocessing'] = lambda seq: [self.SYM_SOS] + seq + [self.SYM_EOS]
        else:
            kwargs['preprocessing'] = lambda seq: [self.SYM_SOS] + seq

        self.sos_id = None
        self.eos_id = None
        super(TargetField, self).__init__(**kwargs)

    def build_vocab(self, *args, **kwargs):
        super(TargetField, self).build_vocab(*args, **kwargs)
        self.sos_id = self.vocab.stoi[self.SYM_SOS]
        self.eos_id = self.vocab.stoi[self.SYM_EOS]
