from machine.util.log import LogCollection
from get_iterations import find_best_its

import numpy as np

from collections import defaultdict

def name_parser(filename, subdir):
    splits = filename.split('/')
    return splits[1]+'_'+splits[2]+'_'+splits[-2]

def data_name_parser(filename):
    splits = filename.split('/')
    data_name = splits[-1][:-4]
    return data_name

def baseline_prernn(modelname):
    if 'baseline' in modelname and 'pre' in modelname:
        return True
    return False

def prernn(modelname):
    if 'pre' in modelname and 'baseline' not in modelname \
            and 'hard' not in modelname:
        return True
    return False

def group_runs_and_samples(model):
    basename = ' '.join(model.split('_')[2:])
    return basename

def get_all_accuracies(log, acc_name, datasets, data_name_parser, best_its):

    accuracies = {}

    for i, log_name in enumerate(log.log_names):
        accuracies[log_name] = {}
        cur_log = log.logs[i]

        # decide iteration to pick
        index = cur_log.steps.index(best_its[log_name])

        for dataset in cur_log.data.keys():
            short_data_name = data_name_parser(dataset)
            if short_data_name in datasets:
                accuracies[log_name][short_data_name] = \
                        cur_log.data[dataset][acc_name][index]

    return accuracies

def average_accuracies(accuracies, group_models):
    av_accs = defaultdict(lambda: defaultdict(list))

    for model in accuracies:
        model_base = group_models(model)
        for dataset, val in accuracies[model].items():
            av_accs[model_base][dataset].append(val)

    for model, model_accs in av_accs.items():
        for dataset in model_accs:
            d = av_accs[model][dataset]
            av_accs[model][dataset] = (np.mean(d), np.std(d), np.min(d), np.max(d))
    return av_accs

def write_to_csv(accs, filename, model_parser, model_write):
    f = open(filename, 'wb')
    f.write('model, compName, seqacc, std, min, max\n')
    for model in accs:
        if model_parser(model):
            for dataset, acc in av_accs[model].items():
                f.write('%s, %s, %f, %f, %f, %f\n' % (model_write, dataset.replace('_', ' '), acc[0], acc[1], acc[2], acc[3]))
    f.close()

# create log
log = LogCollection()
log.add_log_from_folder('chosens_dump', ext='.dump', name_parser=name_parser)

# find best iterations for ever file in log (given file with model names
best_its = find_best_its('chosens_dump/tree.txt')

datasets = ['heldout_inputs', 'heldout_compositions', \
            'heldout_tables', 'new_compositions']

accs = get_all_accuracies(log, 'seq_acc', datasets, data_name_parser, best_its)
av_accs = average_accuracies(accs, group_runs_and_samples)
write_to_csv(av_accs,  'pre_rnn_results_learned.csv', prernn, 'Guided')
# write_to_csv(av_accs,  'pre_rnn_results_baseline.csv', baseline_prernn, 'Baseline')

#for model in av_accs:
#    m_accs = av_accs[model]
#    m_short = '_'.join(model.split('_')[:-1])
#    print('%s:\t\t%s' % (m_short, '\t'.join(['%s %.2f' % (m, m_accs[m]) for m in m_accs])))
