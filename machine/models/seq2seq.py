import torch.nn.functional as F

from .baseModel import BaseModel


class Seq2seq(BaseModel):
    """ Standard sequence-to-sequence architecture with configurable encoder
    and decoder.
    """

    def __init__(self, encoder, decoder, decode_function=F.log_softmax):
        super(Seq2seq, self).__init__(encoder_module=encoder,
                                      decoder_module=decoder, decode_function=decode_function)

    def flatten_parameters(self):
        """
        Flatten parameters of all components in the model.
        """
        self.encoder_module.rnn.flatten_parameters()
        self.decoder_module.rnn.flatten_parameters()

    def forward(self, inputs, input_lengths=None, targets={},
                teacher_forcing_ratio=0):
        # Unpack target variables
        target_output = targets.get('decoder_output', None)

        encoder_outputs, encoder_hidden = self.encoder_module(
            inputs, input_lengths=input_lengths)
        result = self.decoder_module(inputs=target_output,
                                     encoder_hidden=encoder_hidden,
                                     encoder_outputs=encoder_outputs,
                                     function=self.decode_function,
                                     teacher_forcing_ratio=teacher_forcing_ratio)
        return result
