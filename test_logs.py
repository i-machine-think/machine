from seq2seq.util.log import LogCollection
import re

def name_parser(filename, subdir):
    splits = filename.split('/')
    return splits[1]+'_'+splits[-2]


############################
# helper funcs

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


def func(input_str):
    if 'full_focus' in input_str and 'hard' not in input_str and 'baseline' not in input_str:
        return True
    return False

def f64_256(input_str):
    if 'E64xH128' in input_str and 'run_1' in input_str:
        return True
    return False

def pre_rnn(input_str):
    if 'pre_rnn' in input_str\
            and 'baseline' not in input_str and 'hard' not in input_str:
        return True
    return False

def full_focus(input_str):
    if 'full_focus' in input_str\
            and 'baseline' not in input_str and 'hard' not in input_str:
            # and 'E64xH512' in input_str:
        return True
    return False

def pre_ff_baseline(input_str):
    if 'hard' not in input_str:
        return True
    return False

def ff_and_baseline(input_str):
    if ('focus' in input_str and 'baseline' in input_str) or \
            ('focus' in input_str and 'hard' not in input_str):
        return True
    return False

def pre_and_baseline(input_str):
    if 'pre_rnn' in input_str and 'hard' not in input_str\
            and 'H16' not in input_str and 'H32' not in input_str:
        return True
    return False

def best_pre_and_baseline(input_str):
    if 'pre_rnn' in input_str and ( \
            ('hard' not in input_str and 'E16xH512' in input_str \
            and 'baseline' not in input_str) \
            or ('baseline' in input_str and 'E128xH512' in input_str)):
        return True
    return False

def hard(input_str):
    if 'hard' in input_str and 'pre_rnn' in input_str:
        return True
    return False

def baseline(model):
    if 'baseline' in model and 'pre_rnn' in model:
            # and ('E16xH256' in model or 'E62xH256'  in model or 'E64xH512' in model):
        return True
    return False

def data_name_parser(data_name, model_name):
    if 'Train' in data_name and 'baseline' in model_name:
        label = 'Baseline, training loss'
    elif 'Train' in model_name:
        label = 'Attention Guidance, Train'
    elif 'baseline' in model_name:
        label = 'Baseline, test loss'
    else:
        label = 'Attention Guidance, test loss'
    return label

def heldout_tables(input_str):
    if 'heldout_tables' in input_str:
        return True
    return False

def heldout_inputs(input_str):
    if 'heldout_inputs' in input_str:
        return True
    return False

def heldout_compositions(input_str):
    if 'heldout_compositions' in input_str:
        return True
    return False

def not_longer(input_str):
    if 'longer' not in input_str:
        return True
    return False

def not_train(dataset):
    if 'Train' not in dataset:
        return True
    return False

def color_train(model_name, data_name):
    if 'Train' in data_name and 'baseline' in model_name:
        c = 'k--'
    elif 'Train' in data_name:
        c = 'k'
    elif 'baseline' in model_name:
        c = 'm:'
    else:
        c = 'g'

    return c

def color_groups(model_name, data_name):
    if 'baseline' in model_name:
        c = 'b'
    elif 'hard' in model_name:
        c = 'm'
    else:
        c = 'g'

    if 'pre_rnn' in model_name:
        l = ':'
    elif 'full_focus' in model_name:
        l = '-'
    elif 'post_rnn' in model_name:
        l = '--'

    return c+l

def find_basename(model_name):
    all_parts = model_name.split('_')
    basename = '_'.join(all_parts[2:])
    return basename

def no_basename(model_name):
    return model_name

def find_data_name(dataset):
    dataname = dataset.split('/')[-1].split('.')[0]
    if 'longer' in dataname:
        splits = dataname.split('_')
        elements = [splits[0],splits[2]]
        dataname = '_'.join(elements)
    return dataname

def color_baseline(model_name, data_name):
    if 'baseline' in model_name:
        c = 'm'
    else:
        c = 'g'
    return c

def color_conditions(model_name, data_name):
    if 'baseline' in model_name:
        c = 'm'
    elif 'focus' in model_name:
        c = 'b'

    if 'Train' in data_name:
        c = 'k'
        l = '-'
    elif 'inputs' in data_name:
        l = '-'
    elif 'tables' in data_name:
        l = '--'
    elif 'compositions' in data_name and 'heldout' in data_name:
        l = '-.'
    elif 'new' in data_name:
        l = ':'

    return c+l

def color_size(model_name, data_name):
    if 'H16' in model_name:
        c = 'b'
    elif 'H32' in model_name:
        c = 'g'
    elif 'H64' in model_name:
        c = 'k'
    elif 'H128' in model_name:
        c = 'r'
    elif 'H256' in model_name:
        c = 'm'
    elif 'H512' in model_name:
        c = 'c'
    return c


# max_averages = log.find_highest_average('seq_acc', find_basename=no_basename, find_data_name=find_data_name, restrict_data=not_longer, restrict_model=baseline)
# 
# for model in natural_sort(max_averages):
#     datadict = max_averages[model]
#     print('%s:\t%s' % (model, '\t'.join(['%s %.2f' % (d, datadict[d]) for d in datadict])))
# 
# log.plot_metric('seq_acc', restrict_model=full_focus, restrict_data=not_longer)

def plot_size_correlation():
    fig = log.plot_metric('seq_acc', restrict_model=baseline, restrict_data=heldout_tables, data_name_parser=data_name_parser, color_group=color_size, eor=400)


def plot_pre_and_baseline():
    # plot accuracy of all validation sets for best configuration for learned
    # attention (pre) and baseline models to show overfitting
    fig = log.plot_metric('nll_loss', restrict_model=best_pre_and_baseline, restrict_data=not_longer, data_name_parser=data_name_parser, color_group=color_train, eor=-165)
    fig.savefig('/home/dieuwke/Documents/papers/AttentionGuidance/figures/best_config_all_sets_loss.png')

def plot_val_loss():
    # plot accuracy of all validation sets for best configuration for learned
    # attention (pre) and baseline models to show overfitting
    fig = log.plot_metric('nll_loss', restrict_model=best_pre_and_baseline, restrict_data=heldout_tables, data_name_parser=data_name_parser, color_group=color_train)
    # fig.savefig('/home/dieuwke/Documents/papers/AttentionGuidance/figures/best_config_all_sets_loss.png')

def plot_heldout_tables_all():
    # all_models_heldout_tables
    fig = log.plot_metric('seq_acc', restrict_model=pre_and_baseline, restrict_data=heldout_tables, color_group=color_baseline, eor=-165)
    fig.savefig('/home/dieuwke/Documents/papers/AttentionGuidance/figures/all_models_heldout_tables.png')


log = LogCollection()
log.add_log_from_folder('chosens_dump', ext='.dump', name_parser=name_parser)

fig = log.plot_groups('nll_loss', restrict_model=ff_and_baseline, find_basename=find_basename, find_data_name=find_data_name, restrict_data=not_longer, color_group=color_conditions, eor=-135)
fig.savefig('/home/dieuwke/Documents/papers/AttentionGuidance/figures/lookup_loss_convergence.png')
# plot_pre_and_baseline()

# plot_heldout_tables_all()
# plot_size_correlation()
# plot_val_loss()
