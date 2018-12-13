import re

# get the correct iteration for every model, based on validation accuracy


def find_best_its(filename):
    f = open('chosens_dump/tree.txt', 'rb')
    best_iterations = {}

    for line in f:
        if 'sample' in line:
            # basename = line[:-2]
            lsplit = line.split('/')
            cur_model = lsplit[0] + '_' + lsplit[1] + '_' + lsplit[-1][:-2]
            best_acc = 0
        elif 'acc' in line:
            llist = line.split('_')
            seq_acc = llist[3]
            if seq_acc >= best_acc:
                best_acc = seq_acc
                it = int(re.findall(r'\d+', llist[-1])[0])
                # m_name = line[:-1]
        elif 'dump' in line:
            best_iterations[cur_model] = it
            # print basename+'/'+m_name
        else:
            continue
    return best_iterations


best_its = find_best_its('chosens_dump/tree.txt')
