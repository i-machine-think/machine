# -*- coding: utf-8 -*-

"""
Function to generate of the lookup tables problem.

Running this script will save the following files in LookupTables/data/:
- train.tsv
- train_before_new_tables.tsv
- validation.tsv
- heldout_inputs.tsv
- heldout_compositions
- heldout_tables
- new_compositions
- longer_seen_1.tsv
- longer_incremental_1.tsv
- longer_new_1.tsv
...
- longer_seen_n.tsv
- longer_incremental_n.tsv
- longer_new_n.tsv

with n is the max number of additional compositions in test compared to train.

Contact: Yann Dubois / Gautier Dagan
"""

import sys
import os
import ast
import itertools
import random
import operator
import warnings
import argparse
import tqdm
from functools import reduce
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def make_long_lookup_tables(data_folder_path, generation_arguments_path):
    """
    data_folder_path - (path) path of where the data folder (inc. /data) should be i.e. LookupTables/LongLookupTables/data
    generation_arguments_path - (path) path of where the generation_arguments.txt is, i.e. LookupTables/LongLookupTables/generation_arguments.txt
    """
    args = _load_arguments(generation_arguments_path)

    random.seed(args['seed'])
    out = table_lookup_dataset(validation_size=args['validation_size'],
                               max_composition_train=args['max_composition_train'],
                               n_unary_tables=args['n_unary_tables'],
                               n_heldout_tables=args['n_heldout_tables'],
                               n_heldout_compositions=args['n_heldout_compositions'],
                               n_heldout_inputs=args['n_heldout_inputs'],
                               add_composition_test=args['n_longer'],
                               is_reverse=args['reverse'],
                               is_copy_input=not args['not_copy_input'],
                               is_intermediate=not args['not_intermediate'],
                               is_shuffle=not args['not_shuffle'],
                               is_stratify=not args['not_stratify'],
                               is_target_attention=args['is_target_attention'],
                               eos=args['eos'],
                               bound_test=args['bound_test'],
                               seed=args['seed'],
                               alphabet=args['alphabet'],
                               n_repeats=args['n_repeats'],
                               random_start_token=args['random_start_token'],
                               max_noise_tables=args['max_noise_tables'],
                               is_multiple_start_token=args['is_multiple_start_token'],
                               n_intermediate_noise=args['n_intermediate_noise'])

    names = ("train", "train_before_new_tables", "validation", "heldout_inputs",
             "heldout_compositions", "heldout_tables", "new_compositions",
             "longer_seen", "longer_incremental", "longer_new")

    for data, name in zip(out, names):
        _save_tsv(data, name, data_folder_path)


def table_lookup_dataset(validation_size=0.11,
                         max_composition_train=2,
                         n_unary_tables=8,
                         n_heldout_tables=2,
                         n_heldout_compositions=8,
                         n_heldout_inputs=2,
                         add_composition_test=1,
                         is_reverse=False,
                         is_copy_input=True,
                         is_intermediate=True,
                         is_shuffle=True,
                         is_stratify=True,
                         is_target_attention=False,
                         eos=".",
                         bound_test=10**4,
                         random_start_token="!",
                         max_noise_tables=None,
                         is_multiple_start_token=False,
                         n_intermediate_noise=0,
                         seed=123,
                         **kwargs):
    """Prepare the table lookup dataset.

    Args:
        validation_size (float, optional): max length of compositions in training
            set.
        max_composition_train (int, optional): max length of compositions in
            training set.
        n_unary_tables (int, optional): number of different lookup tables.
        n_heldout_tables (int, optional): number of tables that would only be
            seen in unary.
        n_heldout_compositions (int, optional): number of compositions of len
            `max_composition_train` to heldout from training.
        n_heldout_inputs (int, optional): the number of inputs of tables of len
            `max_composition_train` to heldout from training.
        add_composition_test (int, optional): additional composition to add for
            the `longer_*` test data. Those test sets will then include compositions
            between `max_composition_train` and `max_composition_train +
            add_composition_test` tables.
        is_reverse (bool, optional): whether to reverse the  input sequence to
            match the mathematical composition. I.e if given, then uses
            `t1(t2(input))` without parenthesis instead of `input t2 t1`.'
        is_copy_input (bool, optional): whether to include a copy of the initial
            input results in the output.
        is_intermediate (bool, optional): whether to include intermediate results
            in the output.
        is_shuffle (bool, optional): whether to shuffle the outputed datasets.
        is_stratify (bool, optional): whether to split validation to approximately
            balance each lookup table. `validation_size` may have to be larger
            when using this.
        is_target_attention (bool, optional): whether to append the target
            attention as an additional column.
        eos (str, optional): token to append at the end of each input.
        bound_test (int, optional): bounds the number of rows in each test files.
        random_start_token (str, optional): token to be randomly inserted somewhere
            in the sequence to separate noise tables with the ones that you
            should use for the task.
        max_noise_tables (int, optional): maximum number of noise tables to add
            before the `random_start_token`. If `None` then will not insert a
            `random_start_token` and any noise tables.
        is_multiple_start_token (bool, optional): whether to add multiple
            `random_start_token`. If `True` should only start predciting at the
            last occurence.
        n_intermediate_noise (int, optional): Number of intermediate noisy table
            to insert between 2 "real tables".
        seed (int, optional): sets the seed for generating random numbers.
        kwargs: Additional arguments to `create_N_table_lookup`.

    Returns:
        train (pd.Series): dataframe of all multiary training examples. Contains
            all the unary functions. The index is the input and value is the target.
        validation (pd.Series): dataframe of all multiary examples use for validation.
        heldout_inputs (pd.Series): dataframe of inputs that have not been seen
            during training but the mapping have.
        heldout_compositions (pd.Series): dataframe of multiary composition that
            have never been seen during training.
        heldout_tables (pd.Series): dataframe of multiary composition that are
            made up of one table that has never been seen in any multiary
            composition during training.
        new_compositions (pd.Series): dataframe of multiary composition that are
            made up of 2 tables that have never been seen in any multiary
            composition during training.
        longer_seens (list of pd.Series): list of len `add_composition_test`.
            Where the ith element is a dataframe composed of `max_composition_train+i`
            tables that have all been composed in the training set.
        longer_incrementals (list of pd.Series): list of len `add_composition_test`.
            Where the ith element is a dataframe composed of `max_composition_train+i`
            tables, exactly one table that has not been composed in the training set.
        longer_news (list of pd.Series): ist of len `add_composition_test`. Where
            the ith element is a dataframe composed of `max_composition_train+i`
            tables that have never been composed in the training set.
    """
    assert " " not in eos, "Cannot have spaces in the <eos> token."
    if not is_copy_input and is_target_attention:
        raise NotImplementedError(
            "`is_target_attention` with `is_copy=False` not implemented yet.")

    is_noise = (max_noise_tables is not None) or (n_intermediate_noise != 0)

    if is_noise and is_target_attention:
        raise NotImplementedError(
            "noise with `is_target_attention` not implemented yet.")

    if is_reverse and is_noise:
        raise NotImplementedError(
            "`is_reverse` with noise not implemented yet.")

    np.random.seed(seed)
    random.seed(seed)

    unary_functions = create_N_table_lookup(
        N=n_unary_tables, seed=seed, **kwargs)
    n_inputs = len(unary_functions[0])
    names_unary_train = {t.name for t in unary_functions[:-n_heldout_tables]}
    names_unary_test = {t.name for t in unary_functions[-n_heldout_tables:]}
    multiary_functions = [[reduce(lambda x, y: compose_table_lookups(x,
                                                                     y,
                                                                     is_intermediate=is_intermediate),
                                  fs)
                           for fs in itertools.product(unary_functions, repeat=repeat)]
                          for repeat in range(2, max_composition_train + 1)]
    longest_multiary_functions = multiary_functions[-1]
    multiary_functions = flatten(multiary_functions[:-1])
    (longest_multiary_train,
     heldout_tables,
     new_compositions) = _split_seen_unseen_new(longest_multiary_functions,
                                                names_unary_train,
                                                names_unary_test)
    multiary_train, _, _ = _split_seen_unseen_new(multiary_functions,
                                                  names_unary_train,
                                                  names_unary_test)
    random.shuffle(longest_multiary_train)

    # heldout
    heldout_compositions = longest_multiary_train[-n_heldout_compositions:]

    longest_multiary_train = longest_multiary_train[:-n_heldout_compositions]
    drop_inputs = [np.random.choice(table.index, n_heldout_inputs, replace=False)
                   for table in longest_multiary_train]
    heldout_inputs = [table[held_inputs]
                      for held_inputs, table in zip(drop_inputs, longest_multiary_train)]

    longest_multiary_train = [table.drop(held_inputs)
                              for held_inputs, table in zip(drop_inputs,
                                                            longest_multiary_train)]

    # longer
    longer_seens = []
    longer_incrementals = []
    longer_news = []
    longer = [compose_table_lookups(x, y)
              for x, y in itertools.product(unary_functions, longest_multiary_functions)]
    for _ in range(add_composition_test):
        (longer_seen,
         longer_incremental,
         longer_new) = _split_seen_unseen_new(longer,
                                              names_unary_train,
                                              names_unary_test)

        for longer_i, longer_i_list in zip([longer_seen, longer_incremental, longer_new],
                                           [longer_seens, longer_incrementals, longer_news]):
            # uses round(bound_test/n_inputs) because at that moment we have a
            # list of composed tables with each `n_inputs` rows. At the end we
            # will merge those and bound_test should filter the total number of rows.
            if len(longer_i) * n_inputs > bound_test:
                mssg = "Randomly select tables as len(longer)={} is larger than bound_test={}."
                warnings.warn(mssg.format(
                    n_inputs * len(longer_i), bound_test))
                longer_i = random.sample(
                    longer_i, round(bound_test / n_inputs))

            longer_i_list.append(longer_i)

        longer = flatten(
            [longer_seens[-1], longer_incrementals[-1], longer_news[-1]])
        longer = [compose_table_lookups(x, y)
                  for x, y in itertools.product(unary_functions, longer)]

    # formats
    longer_seens = _merge_format_inputs(longer_seens, is_shuffle,
                                        bound_test=bound_test,
                                        seed=seed,
                                        is_reverse=is_reverse,
                                        eos=eos)
    longer_incrementals = _merge_format_inputs(longer_incrementals, is_shuffle,
                                               bound_test=bound_test,
                                               seed=seed,
                                               is_reverse=is_reverse,
                                               eos=eos)
    longer_news = _merge_format_inputs(longer_news, is_shuffle,
                                       bound_test=bound_test,
                                       seed=seed,
                                       is_reverse=is_reverse,
                                       eos=eos)

    multiary_train += longest_multiary_train
    building_blocks = [multiary_train, heldout_inputs, heldout_compositions,
                       heldout_tables, new_compositions]
    # don't shuffle unary_functions becaue will be separating the train and test
    # for those by indexing
    unary_functions_list = _merge_format_inputs([unary_functions], False,
                                                bound_test=None,
                                                seed=seed,
                                                is_reverse=is_reverse,
                                                eos=eos)
    # don't bound test because size check after
    building_blocks = _merge_format_inputs(building_blocks, is_shuffle,
                                           bound_test=None,
                                           seed=seed,
                                           is_reverse=is_reverse,
                                           eos=eos)
    building_blocks = unary_functions_list + building_blocks
    _check_sizes(building_blocks, n_inputs, max_composition_train, n_unary_tables,
                 n_heldout_tables, n_heldout_compositions, n_heldout_inputs)
    if bound_test is not None:
        # bound only testing sets
        building_blocks[2:] = [df.iloc[:bound_test]
                               for df in building_blocks[2:]]
    (unary_functions,
     multiary_train,
     heldout_inputs,
     heldout_compositions,
     heldout_tables,
     new_compositions) = building_blocks

    # validation
    multiary_train, validation = _uniform_split(multiary_train, names_unary_train,
                                                validation_size=validation_size,
                                                seed=seed)

    train_all = pd.concat([unary_functions, multiary_train], axis=0)
    unary_functions_train = unary_functions.iloc[:len(
        names_unary_train) * n_inputs]
    train_before_new_tables = pd.concat(
        [unary_functions_train, multiary_train], axis=0)

    out = (train_all, train_before_new_tables, validation, heldout_inputs,
           heldout_compositions, heldout_tables, new_compositions, longer_seens,
           longer_incrementals, longer_news)

    # add noise tables
    if max_noise_tables is not None:
        for o in out:
            _add_noise_tables(o,
                              names_unary_train,
                              max_noise_tables=max_noise_tables,
                              random_start_token=random_start_token,
                              is_multiple_start_token=is_multiple_start_token)

    if n_intermediate_noise != 0:
        for o in out:
            _add_intermediate_noise_tables(o,
                                           names_unary_train,
                                           n_intermediate_noise=n_intermediate_noise)

    # adds target attention
    if is_target_attention:
        out = [_append_target_attention(o, eos, is_reverse) for o in out[:-3]]
        for longer in (longer_seens, longer_incrementals, longer_news):
            out.append([_append_target_attention(o, eos, is_reverse)
                        for o in longer])

    return out


def create_N_table_lookup(N=None,
                          alphabet=['0', '1'],
                          n_repeats=3,
                          namer=lambda i: "t{}".format(i + 1),
                          seed=123):
    """Create N possible table lookups.

    Args:
        N (int, optional): number of tables lookups to create. (default: all possible)
        alphabet (list of char, optional): possible characters given as input.
        n_repeats (int, optional): number of characters in `alphabet` used in each
            input and output.
        namer (callable, optional): function that names a table given an index.
        seed (int, optional): sets the seed for generating random numbers.

    Returns:
        out (list of pd.Series): list of N dataframe with keys->input, data->output,
            name->namer(i).
    """
    np.random.seed(seed)
    inputs = np.array(list(''.join(letters)
                           for letters in itertools.product(alphabet, repeat=n_repeats)))
    iter_outputs = itertools.permutations(inputs)
    if N is not None:
        iter_outputs = np.array(list(iter_outputs))
        indices = np.random.choice(range(len(iter_outputs)),
                                   size=N,
                                   replace=False)
        iter_outputs = iter_outputs[indices]
    return [pd.Series(data=outputs, index=inputs, name=namer(i))
            for i, outputs in enumerate(iter_outputs)]


def compose_table_lookups(table1, table2, is_intermediate=True):
    """Create a new table lookup as table1 ∘ table2."""
    left = table1.to_frame()
    right = table2.to_frame()
    right['next_input'] = right.iloc[:, 0].str.split().str[-1]
    merged_df = pd.merge(left, right, left_index=True, right_on='next_input'
                         ).drop("next_input", axis=1)
    left_col, right_col = merged_df.columns

    if is_intermediate:
        merged_serie = merged_df[right_col] + " " + merged_df[left_col]
    else:
        merged_serie = merged_df[left_col]

    merged_serie.name = " ".join(
        [left_col.split("_")[0], right_col.split("_")[0]])

    return merged_serie


def format_input(table, is_copy_input=True, is_reverse=False, eos=None):
    """Formats the input of the task.

    Args:
        table (pd.Series, optional): Serie where keys->input, data->output,
            name->namer(i)
        is_copy_input (bool, optional): whether to have the inputs first and then
            the tables. Ex: if reverse: 001 t1 t2 else t2 t1 001.
        is_reverse (bool, optional): whether to reverse the  input sequence to
            match the mathematical composition. I.e if given, then uses
            `t1(t2(input))` without parenthesis instead of `input t2 t1`.'
        eos (str, optional): str to append at the end of each input.

    Returns:
        out (pd.Series): Serie where keys->input+name, data->output, name->namer(i).
    """
    inputs = table.index

    table.index = ["{} {}".format(table.name, i) for i in table.index]

    if not is_reverse:
        table.index = [" ".join(i.split()[::-1]) for i in table.index]

    if eos is not None:
        table.index = ["{} {}".format(i, eos) for i in table.index]

    if is_copy_input:
        table.iloc[:] = inputs + " " + table

    return table


### HELPERS ###
def _save_arguments(args, directory, filename="generation_arguments.txt"):
    """Save arguments to a file given a dictionnary."""
    with open(os.path.join(directory, filename), 'w') as file:
        file.writelines('{}={}\n'.format(k, v) for k, v in args.items())


def _load_arguments(argument_path):
    """Load arguments to a dictionnary (reverse of above operation)."""
    args = {}
    with open(argument_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            l = line.split("=")
            args[l[0]] = l[1][:-1]
    args['n_samples'] = int(args['n_samples'])
    args['validation_size'] = float(args['validation_size'])
    args['max_composition_train'] = int(args['max_composition_train'])
    args['n_unary_tables'] = int(args['n_unary_tables'])
    args['n_heldout_tables'] = int(args['n_heldout_tables'])
    args['n_heldout_compositions'] = int(args['n_heldout_compositions'])
    args['n_heldout_inputs'] = int(args['n_heldout_inputs'])
    args['n_longer'] = int(args['n_longer'])
    args['n_repeats'] = int(args['n_repeats'])
    args['seed'] = int(args['seed'])
    args['bound_test'] = int(args['bound_test'])
    args['n_intermediate_noise'] = int(args['n_intermediate_noise'])

    args['reverse'] = args['reverse'] == "True"
    args['not_copy_input'] = args['not_copy_input'] == "True"
    args['not_shuffle'] = args['not_shuffle'] == "True"
    args['not_stratify'] = args['not_stratify'] == "True"
    args['is_target_attention'] = args['is_target_attention'] == "True"
    args['is_multiple_start_token'] = args['is_multiple_start_token'] == "True"
    args['not_intermediate'] = args['not_intermediate'] == "True"

    args['max_noise_tables'] = (
        None if args['max_noise_tables'] == "None" else int(args['max_noise_tables']))

    args['alphabet'] = ast.literal_eval(args['alphabet'])
    args['alphabet'] = [x.strip() for x in args['alphabet']]

    return args


def _save_tsv(data, name, path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

    if isinstance(data, list):
        for i, df in enumerate(data):
            df.to_csv(os.path.join(path, "{}_{}.tsv".format(name, i + 1)),
                      header=False,
                      sep=str('\t'))  # wraps sep around str for python 2
    else:
        data.to_csv(os.path.join(path, "{}.tsv".format(name)),
                    header=False,
                    sep=str('\t'))  # wraps sep around str for python 2


def flatten(l):
    """flattens a list."""
    if l == []:
        return []
    elif isinstance(l, list):
        return reduce(operator.add, l)
    else:
        return l


def intertwine(l1, l2):
    """Intertwines 2 list such that the final list is [l1[0], l2[0],
    ..., l1[-1], l2[-1]].
    """
    assert (len(l1) == len(l2)) or (len(l1) == len(l2) + 1)
    if len(l1) == len(l2) + 1:
        return [item for pair in zip(l1, l2) for item in pair] + [l1[-1]]
    return [item for pair in zip(l1, l2) for item in pair]


def assert_equal(a, b):
    assert a == b, "{} != {}".format(a, b)


def _split_seen_unseen_new(dfs, name_train, name_test):
    """
    Split list of datatframes such that `seen` has only tables in `name_train`,
    `new` has only tables in `name_test`, and `unseen` has exactly one table in
    `name_test`.
    """

    def _table_is_composed_of(composed_table, tables, n_tables=None):
        if n_tables is None:
            # compsed of at least one
            return set(composed_table.name.split()).intersection(tables)
        else:
            # at least one table composed of exactly `n_tables`
            return sum(Counter(composed_table.name.split())[table]
                       for table in tables) == n_tables

    seen = [t for t in dfs if not _table_is_composed_of(t, name_test)]
    new = [t for t in dfs if not _table_is_composed_of(t, name_train)]
    unseen = [t for t in dfs
              if (_table_is_composed_of(t, name_test, n_tables=1) and
                  _table_is_composed_of(t, name_train))]
    return seen, unseen, new


def _merge_format_inputs(list_dfs, is_shuffle, bound_test=None, seed=None, **kwargs):
    if list_dfs == []:
        return []

    list_df = [pd.concat([format_input(df, **kwargs) for df in dfs],
                         axis=0)
               if dfs != [] else pd.DataFrame()
               for dfs in list_dfs]

    if is_shuffle:
        list_df = [df.sample(frac=1, random_state=seed) for df in list_df]

    if bound_test is not None:
        # better to use is_shuffle when bounding test
        list_df = [df.iloc[:bound_test] for df in list_df]

    return list_df


def _uniform_split(to_split, table_names, validation_size=0.1, seed=None, is_stratify=True):
    df = to_split.to_frame()
    for name in table_names:
        df[name] = [name in i.split() for i in df.index]
    df['length'] = [len(i.split()) for i in df.index]

    stratify = df.iloc[:, 1:] if is_stratify else None

    try:
        train, test = train_test_split(to_split,
                                       test_size=validation_size,
                                       random_state=seed,
                                       stratify=stratify)
    except ValueError:
        warnings.warn(
            "Doesn't use stratfy as given validation_size was to small.")
        train, test = train_test_split(to_split,
                                       test_size=validation_size,
                                       random_state=seed,
                                       stratify=None)

    return train, test


def _check_sizes(dfs, n_inputs, max_length, n_unary_tables, n_heldout_tables,
                 n_heldout_compositions, n_heldout_inputs):
    (unary_functions, multiary_train, heldout_inputs, heldout_compositions,
        heldout_tables, new_compositions) = dfs

    # n_inputs is alphabet**n_repeats
    n_train_tables = n_unary_tables - n_heldout_tables
    n_train_compositions = sum(n_train_tables**i
                               for i in range(2, max_length + 1)) - n_heldout_compositions

    def _size_permute_compose(n_tables):
        return sum(n_tables**i * n_inputs for i in range(2, max_length + 1))

    def _size_compose(n_tables, max_length=max_length):
        return n_tables**max_length * n_inputs

    assert_equal(len(unary_functions),
                 n_unary_tables * n_inputs)
    assert_equal(len(heldout_inputs),
                 (n_train_tables**max_length - n_heldout_compositions) * n_heldout_inputs)
    assert_equal(len(multiary_train),
                 n_train_compositions * n_inputs - len(heldout_inputs))
    assert_equal(len(heldout_compositions),
                 n_heldout_compositions * n_inputs)
    assert_equal(len(heldout_tables),
                 _size_compose(n_train_tables, max_length - 1) * max_length * 2)
    assert_equal(len(new_compositions),
                 _size_compose(n_heldout_tables))


def _append_target_attention(df, eos, is_reverse):
    """Appends the target attention by returning a datafarme with the attention
    given a series.
    """
    def _len_no_eos(s):
        return len([el for el in s.split() if el != eos])

    df = df.to_frame()
    df["taget attention"] = [" ".join(str(i) for i in range(_len_no_eos(inp)))
                             for inp in df.index]
    if is_reverse:
        df["taget attention"] = [ta[::-1] for ta in df["taget attention"]]
    if eos != "":
        df["taget attention"] = [ta + " " + str(len(ta.split()))
                                 for ta in df["taget attention"]]
    return df


def _add_noise_tables(data,
                      names_unary_train,
                      max_noise_tables=10,
                      random_start_token="!",
                      is_multiple_start_token=False):
    """Prepend a uniform number between 0 and `max_noise_tables` of noisy training tables
    at the begining of the source sequences. The end of the noisy tables with be indicates
    by `random_start_token`. With `is_multiple_start_token`, the noisy symbols may include
    `random_start_token`, in which case only the last `random_start_token` will indicate
    the end of the noisy tables.
    """

    def noise(n, possible_noisy_tokens):
        n_start_token = np.random.randint(n) if (
            is_multiple_start_token and n != 0) else 0
        noisy_tokens = random.choices(possible_noisy_tokens, k=(n - n_start_token)
                                      ) + [random_start_token] * n_start_token
        np.random.shuffle(noisy_tokens)
        return noisy_tokens + [random_start_token]

    if isinstance(data, list):
        for d in data:
            _add_noise_tables(d,
                              names_unary_train,
                              max_noise_tables=max_noise_tables,
                              random_start_token=random_start_token,
                              is_multiple_start_token=is_multiple_start_token)

    else:
        possible_noisy_tokens = list(names_unary_train)
        n_noise_tables = np.random.randint(
            max_noise_tables, size=len(data.index))

        noise_tables = [noise(n, possible_noisy_tokens)
                        for n in n_noise_tables]

        data.index = [" ".join([i.split(" ", 1)[0]] + nt + [i.split(" ", 1)[1]])
                      for nt, i in zip(noise_tables, data.index)]


def _add_intermediate_noise_tables(data,
                                   names_unary_train,
                                   n_intermediate_noise=1):
    """Adds `n_intermediate_noise` noisy tables between each of the "real" ones."""
    def add_noise_single(index, possible_noisy_tokens):
        splitted_index = index.split()
        n_noise = len(splitted_index) - 1
        noise = [" ".join(random.choices(possible_noisy_tokens,
                                         k=n_intermediate_noise))
                 for _ in range(n_noise)]
        intertwined = intertwine(splitted_index, noise)
        return " ".join(intertwined)

    if isinstance(data, list):
        for d in data:
            _add_intermediate_noise_tables(d,
                                           names_unary_train,
                                           n_intermediate_noise=n_intermediate_noise)

    else:
        possible_noisy_tokens = list(names_unary_train)

        data.index = [add_noise_single(
            i, possible_noisy_tokens) for i in data.index]
