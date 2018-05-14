from __future__ import print_function

import torch

from collections import defaultdict

class Log(object):
    """
    The Log can be used to store logs during training, write the to a file
    and read them again later.
    """

    def __init__(self):
        self.steps = []
        self.log = defaultdict(lambda: defaultdict(list))

    def write_to_log(self, dataname, losses, metrics, step):
        """
        Add new losses to Log object.
        """
        for metric in metrics:
            val = metric.get_val()
            self.log[dataname][metric.log_name].append(val)

        for loss in losses:
            val = loss.get_loss()
            self.log[dataname][loss.log_name].append(val)

    def update_step(self, step):
        self.steps.append(step)

    def write_to_file(self, path):
        f = open(path, 'wb')

        # write steps
        f.write("steps %s\n" % ' '.join([str(step) for step in self.steps]))

        # write logs
        for dataset in self.log.keys():
            f.write(dataset+'\n')
            for metric in self.log[dataset]:
                f.write('\t%s %s\n' % (metric, ' '.join([str(v) for v in self.log[dataset][metric]])))

        f.close()

    def read_from_file(self, path):
        f = open(path, 'rb')

        lines = f.readlines()
        self.steps = [int(i) for i in lines[0].split()[1:]]

        for line in lines[1:]:
            l_list = line.split()
            if len(l_list) == 1:
                cur_set = l_list[0]
            else:
                data = [float(i) for i in l_list[1:][1:]]
                self.log[cur_set][l_list[0]] = data

    def get_logs(self):
        return self.log

    def get_steps(self):
        return self.steps

