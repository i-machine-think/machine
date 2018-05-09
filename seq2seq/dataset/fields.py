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
    include_eos = True

    def __init__(self, include_eos=True, **kwargs):
        logger = logging.getLogger(__name__)
        self.include_eos = include_eos

        if kwargs.get('batch_first') == False:
            logger.warning("Option batch_first has to be set to use pytorch-seq2seq.  Changed to True.")
        kwargs['batch_first'] = True

        if kwargs.get('preprocessing') is None:
            func = lambda seq: seq
        else:
            func = kwargs['preprocessing']

        if self.include_eos:
            app_eos = [self.SYM_EOS]
        else:
            app_eos = []

        kwargs['preprocessing'] = lambda seq: [self.SYM_SOS] + func(seq) + app_eos

        self.sos_id = None
        self.eos_id = None
        super(TargetField, self).__init__(**kwargs)

    def build_vocab(self, *args, **kwargs):
        super(TargetField, self).build_vocab(*args, **kwargs)
        self.sos_id = self.vocab.stoi[self.SYM_SOS]
        self.eos_id = self.vocab.stoi[self.SYM_EOS]

class AttentionField(torchtext.data.Field):
    """ Wrapper class of torchtext.data.Field that forces batch_first to be True, use_vocab to be false, and applies postprocessing to integers
    Since we already define the attention vectors with integers in the data set, we don't need a vocabulary. Instead, we directly use the provided integers
    """

    def __init__(self, ignore_index, **kwargs):
        """
        Initialize the AttentionField. As pre-processing it prepends the ignore value, to account for the SOS step
        
        Args:
            ignore_index (int): The value that will be ignored for metric and loss calculation, when using attention loss
            **kwargs: The extra arguments for the parent class 
        """
        logger = logging.getLogger(__name__)

        if kwargs.get('batch_first') == False:
            logger.warning("Option batch_first has to be set to use pytorch-seq2seq. Changed to True.")
        kwargs['batch_first'] = True

        if kwargs.get('use_vocab') == True:
            logger.warning("Option use_vocab has to be set to False for the attention field. Changed to False")
        kwargs['use_vocab'] = False

        if kwargs.get('preprocessing') is not None:
            logger.error("No pre-processing allowed for the attention field")

        def preprocess(seq):
            return [self.ignore_index] + seq

        if kwargs.get('postprocessing') is not None:
            logger.error("No post-processing allowed for the attention field")

        # Post-processing function receives batch and positional arguments(?).
        # Batch is a 2D list with batch examples in dim-0 and sequences in dim-1
        # For each element in each example we convert from unicode string to integer.
        # PAD is converted to -1
        def postprocess(batch, _, __):
            def safe_cast(cast_func, x, default):
                try:
                    return cast_func(x)
                except (ValueError, TypeError):
                    return default

            return [[safe_cast(int, item, self.ignore_index) for item in example] for example in batch]
        
        kwargs['preprocessing'] = preprocess
        kwargs['postprocessing'] = postprocess

        super(AttentionField, self).__init__(**kwargs)

        self.ignore_index = ignore_index
