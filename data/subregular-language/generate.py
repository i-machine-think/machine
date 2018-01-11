import sys
import itertools
import logging
import random
import copy
logging.basicConfig(level=logging.INFO)

show_progress = False
n_grammars = 200
n_examples = 400
K = 3  # maximum size of k-factors
k = 3  # minumum size of k-factors
F = 5  # number of k-factors in a grammar
example_length = 20
alphabet_size = 6
alphabet ="".join(chr(i) for i in range(ord('a'), ord('a') + alphabet_size))

k_fact= list(itertools.chain(*(itertools.combinations_with_replacement(alphabet, i) for i in range(k,K+1))))
k_fact = list(map(lambda x:"".join(x), k_fact))
logging.debug("Set of possible k-factors: {}".format(k_fact))

def gen_grammar():
    g = []
    k_fact_pool = copy.copy(k_fact)
    while len(g) < F:
        if len(k_fact_pool) == 0:
            raise RuntimeError("No more k-factors to choose when generating "
                    "grammar {} with size {}".format(g, F))
        # choose a candidate k-factor
        kf = random.choice(k_fact_pool)
        # add it to the grammar
        g.append(kf)
        # remove from the k-factor pool those that are a superset of the chosen kf
        k_fact_pool = [f for f in k_fact_pool if kf not in f]
    return list(sorted(g))

def belongs(g, s):
    for kf in g:
        if kf in s:
            return False
    return True
    
def gen_example(g, cls):
    # very inefficient solution for some grammars (those that contain very few
    # positive examples)
    done = False
    while not done:
        s = "".join(random.choices(alphabet, k=example_length))
        if belongs(g, s) == cls:
            return s

for i in range(n_grammars):
    g = gen_grammar()
    g_str = " ".join(g)
    for j in range(n_examples):
        if show_progress:
            sys.stderr.write("\r{:02.2f}%".format(100*(i*n_examples+j)/n_grammars/n_examples))
        pos = gen_example(g, True)
        print("\t".join([g_str, pos, "1"]))
        neg = gen_example(g, False)
        print("\t".join([g_str, neg, "0"]))

