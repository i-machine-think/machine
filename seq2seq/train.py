import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
import datetime
import math
import os
import random
import io
import socket
# hostname = socket.gethostname()
hostname = "http://localhost:8888"

import numpy as np
from torch import optim
import torchvision

from models import *
# import data
from data import indexes_from_sentence, random_batch
import Constants

parser = argparse.ArgumentParser(description='train.py')
# **Preprocess Options**
parser.add_argument('-savedata', required=True, type=str,
                    help="Output file for the prepared data")

# **Configure models
parser.add_argument('-attn_model', type=str, default='dot',
                    help='Attention model')
parser.add_argument('-hidden_size', type=int, default=100,
                    help='Size of the hidden states')
parser.add_argument('-n_layers', type=int, default=2,
                    help='Number of layers in the encoder/decoder')
parser.add_argument('-dropout', type=float, default=0.1,
                    help='Dropout probability.')

# **Configure training/optimization
parser.add_argument('-batch_size', type=int, default=5,
                    help='Batch size.')
parser.add_argument('-n_epochs', type=int, default=10000,
                    help='Number of epochs.')
parser.add_argument('-clip', type=float, default=50.0,
                    help='Gradient clipping.')
parser.add_argument('-teacher_forcing_ratio', type=float, default=0.5,
                    help='Teacher/forcing ration.')
parser.add_argument('-learning_rate', type=float, default=0.0001,
                    help='Learning rate.')
parser.add_argument('-decoder_learning_ratio', type=float, default=5.0,
                    help='Decoder learning ratio.')

# **Training plot, save, log & eval
parser.add_argument('-plot', type=bool, default=False,
                    help='Whether plotting training.')
parser.add_argument('-log', type=bool, default=False,
                    help='Whether logging training.')
parser.add_argument('-evaluate', type=bool, default=False,
                    help='Whether evaluating training.')
parser.add_argument('-save', type=bool, default=False,
                    help='Whether saving models.')
parser.add_argument('-plot_every', type=int, default=10,
                    help='When to plot train info.')
parser.add_argument('-print_every', type=int, default=10,
                    help='When to print train info.')
parser.add_argument('-evaluate_every', type=int, default=10,
                    help='When to evaluate the model.')
parser.add_argument('-save_every', type=int, default=100,
                    help='When to save the model.')
parser.add_argument('-save_models_path', type=str, default='models',
                    help='When to save the model.')

# GPU
parser.add_argument('-cuda', action='store_true',
                    help="Use CUDA")

opt = parser.parse_args()

if opt.plot or opt.evaluate:
    from PIL import Image
    import visdom
    vis = visdom.Visdom()

# load data
savedata = torch.load(opt.savedata + '.pt')
vocab_source = savedata['vocab_source']
vocab_target = savedata['vocab_target']
train_pairs = savedata['train_pairs']

def train_model():

    epoch = 0

    # Initialize models
    encoder = EncoderRNN(vocab_source.n_words, opt.hidden_size, opt.n_layers, dropout=opt.dropout)
    decoder = LuongAttnDecoderRNN(opt.attn_model, opt.hidden_size, vocab_target.n_words, opt.n_layers, dropout=opt.dropout)

    # Initialize optimizers and criterion
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=opt.learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=opt.learning_rate * opt.decoder_learning_ratio)
    criterion = nn.CrossEntropyLoss()

    # Move models to GPU
    if opt.cuda:
        encoder.cuda()
        decoder.cuda()

    import sconce
    job = sconce.Job('seq2seq-translate', {
        'attn_model': opt.attn_model,
        'n_layers': opt.n_layers,
        'dropout': opt.dropout,
        'hidden_size': opt.hidden_size,
        'learning_rate': opt.learning_rate,
        'clip': opt.clip,
        'teacher_forcing_ratio': opt.teacher_forcing_ratio,
        'decoder_learning_ratio': opt.decoder_learning_ratio,
    })

    if opt.plot:
        job.plot_every = opt.plot_every
    if opt.log:
        job.log_every = opt.print_every

    # Keep track of time elapsed and running averages
    start = time.time()
    plot_losses = []
    print_loss_total = 0 # Reset every print_every
    plot_loss_total = 0 # Reset every plot_every

    # Begin!
    ecs = []
    dcs = []
    eca = 0
    dca = 0
    while epoch < opt.n_epochs:
        epoch += 1

        # Get training data for this cycle
        input_batches, input_lengths, target_batches, target_lengths = random_batch(opt.batch_size, vocab_source, vocab_target, train_pairs)

        # Run the train function
        loss, ec, dc = train(
            input_batches, input_lengths, target_batches, target_lengths,
            encoder, decoder,
            encoder_optimizer, decoder_optimizer, criterion
        )

        # Keep track of loss
        print_loss_total += loss
        plot_loss_total += loss
        eca += ec
        dca += dc

        job.record(epoch, loss)

        if opt.log and epoch % opt.print_every == 0:
            print_loss_avg = print_loss_total / opt.print_every
            print_loss_total = 0
            print_summary = '%s (%d %d%%) %.4f' % (
            time_since(start, max(1, epoch / opt.n_epochs)), epoch, epoch / opt.n_epochs * 100, print_loss_avg)
            print(print_summary)

        if opt.evaluate and epoch % opt.evaluate_every == 0:
            evaluate_randomly(encoder, decoder, train_pairs)

        if opt.plot and epoch % opt.plot_every == 0:
            plot_loss_avg = plot_loss_total / opt.plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

            # TODO: Running average helper
            ecs.append(eca / opt.plot_every)
            dcs.append(dca / opt.plot_every)
            ecs_win = 'encoder grad (%s)' % hostname
            dcs_win = 'decoder grad (%s)' % hostname
            vis.line(np.array(ecs), win=ecs_win, opts={'title': ecs_win})
            vis.line(np.array(dcs), win=dcs_win, opts={'title': dcs_win})
            eca = 0
            dca = 0

        if opt.save and epoch % opt.save_every == 0:
            timestamp = datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M")
            if not os.path.exists(opt.save_models_path):
                os.makedirs(opt.save_models_path)
            torch.save(encoder.state_dict(), os.path.join(opt.save_models_path, ''.join(
                ['encoder',  timestamp, '_ep', str(epoch)])))
            torch.save(decoder.state_dict(), os.path.join(opt.save_models_path, ''.join(
                ['decoder', timestamp, '_ep', str(epoch)])))

def train(input_batches, input_lengths, target_batches, target_lengths, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion, max_length=Constants.MAX_LENGTH):
    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0  # Added onto for each word

    # Run words through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)

    # Prepare input and output variables
    decoder_input = Variable(torch.LongTensor([Constants.SOS_token] * opt.batch_size))
    decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder

    max_target_length = max(target_lengths)
    all_decoder_outputs = Variable(torch.zeros(max_target_length, opt.batch_size, decoder.output_size))

    # Move new Variables to CUDA
    if opt.cuda:
        decoder_input = decoder_input.cuda()
        all_decoder_outputs = all_decoder_outputs.cuda()

    # Run through decoder one time step at a time
    for t in range(max_target_length):
        decoder_output, decoder_hidden, decoder_attn = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )

        all_decoder_outputs[t] = decoder_output
        decoder_input = target_batches[t]  # Next input is current target

    # Loss calculation and backpropagation
    loss = masked_cross_entropy(
        all_decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq
        target_batches.transpose(0, 1).contiguous(),  # -> batch x seq
        target_lengths
    )
    loss.backward()

    # Clip gradient norms
    ec = torch.nn.utils.clip_grad_norm(encoder.parameters(), opt.clip)
    dc = torch.nn.utils.clip_grad_norm(decoder.parameters(), opt.clip)

    # Update parameters with optimizers
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0], ec, dc

def evaluate(encoder, decoder, input_seq, max_length=Constants.MAX_LENGTH):
    input_lengths = [len(input_seq)]
    input_seqs = [indexes_from_sentence(vocab_source, input_seq)]
    input_batches = Variable(torch.LongTensor(input_seqs), volatile=True).transpose(0, 1)

    if opt.cuda:
        input_batches = input_batches.cuda()

    # Set to not-training mode to disable dropout
    encoder.train(False)
    decoder.train(False)

    # Run through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)

    # Create starting vectors for decoder
    decoder_input = Variable(torch.LongTensor([Constants.SOS_token]), volatile=True)  # SOS
    decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder

    if opt.cuda:
        decoder_input = decoder_input.cuda()

    # Store output words and attention states
    decoded_words = []
    decoder_attentions = torch.zeros(max_length + 1, max_length + 1)

    # Run through decoder
    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )
        decoder_attentions[di, :decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

        # Choose top word from output
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == Constants.EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(vocab_target.index2word[ni])

        # Next input is chosen word
        decoder_input = Variable(torch.LongTensor([ni]))
        if opt.cuda: decoder_input = decoder_input.cuda()

    # Set back to training mode
    encoder.train(True)
    decoder.train(True)

    return decoded_words, decoder_attentions[:di + 1, :len(encoder_outputs)]

def evaluate_randomly(encoder, decoder, train_pairs):
    [input_sentence, target_sentence] = random.choice(train_pairs)
    evaluate_and_show_attention(encoder, decoder, input_sentence, target_sentence)

def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

def show_plot_visdom():
    buf = io.BytesIO()
    plt.savefig(buf)
    buf.seek(0)
    attn_win = 'attention (%s)' % hostname
    vis.image(torchvision.transforms.ToTensor()(Image.open(buf)), win=attn_win, opts={'title': attn_win})

def show_attention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    show_plot_visdom()
    plt.show()
    plt.close()

def evaluate_and_show_attention(encoder, decoder, input_sentence, target_sentence=None):
    output_words, attentions = evaluate(encoder, decoder, input_sentence)
    output_sentence = ' '.join(output_words)
    print('>', input_sentence)
    if target_sentence is not None:
        print('=', target_sentence)
    print('<', output_sentence)

    # todo: fix this!
    # show_attention(input_sentence, output_words, attentions)

    # Show input, target, output text in visdom
    win = 'evaluted (%s)' % hostname
    text = '<p>&gt; %s</p><p>= %s</p><p>&lt; %s</p>' % (input_sentence, target_sentence, output_sentence)
    vis.text(text, win=win, opts={'title': win})

if __name__ == "__main__":
    train_model()