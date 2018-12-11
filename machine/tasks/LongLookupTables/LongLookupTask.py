import os
import json
import logging

from machine.tasks import Task
from machine.tasks.LongLookupTables.make_long_lookup_tables import make_long_lookup_tables
from machine.tasks.utils import get_default_params, flatten, repeat, filter_dict


dir_path = os.path.dirname(os.path.realpath(__file__))

# Dictionary that maps input name of lookup table dataset
# to directory where it is located
name2dir = {
    "long_lookup": "LongLookupTables",
    "long_lookup_oneshot": "LongLookupTablesOneShot",
    "long_lookup_reverse": "LongLookupTablesReverse",
    "long_lookup_intermediate_noise": "LongLookupTablesIntermediateNoise",
    "noisy_long_lookup_single": "NoisyLongLookupTablesSingle",
    "noisy_long_lookup_multi": "NoisyLongLookupTablesMulti"
}


class LongLookupTask (Task):
    """
    Return the wanted lookup dataset information, downloads or generates
    it if it is not already present

    Args:
        name: (str): name of the long lookup task to get
            Implemented Options: 
            "long_lookup" :  Lookup tables with training up to 3 compositions
            "long_lookup_oneshot" : long lookup tables with a iniital training 
                                    file without t7 and t8 and then adding 
                                    uncomposed t7 and t8 with all the rest
            "long_lookup_reverse" : reverse long lookup table (i.e right to left hard attention)
            "noisy_long_lookup_multi" : noisy long lookup table where between each 
                                        "real" table there's one noisy one.
                                        The hard attention is thus a diagonal wich is less steep
            "noisy_long_lookup_single" : noisy long lookup table with a special start token saying
                                         when are the "real tables" starting. The hard attention 
                                         is thus a diagonal that starts at some random position.
            "long_lookup_intermediate_noise" : noisy long lookup table where there are multiple 
                                               start token and only the last one really counts
            NotImplemented Option:
            "long lookup jump" : Lookup tables with training 1 2 and 4 compositions (i.e jumping 3) 
                                 currently does not have appropriate generation.txt

        is_small (bool, optional): whether to run a smaller verson of the task.
            Used for getting less statistically significant results.
        is_mini (bool, optional): whether to run a smaller verson of the task.
            Used for testing purposes.
        longer_repeat (int, optional): number of longer test sets. 
            - note if data is already generated with a certain longer repeat 
            - If a longer repeat is called then the return paths will not exist
            - In this case either delete the data folder from the specific LongLookup set
            - Or implement a check to extend it and regenerate with higher longer_repeat
        logger (object, optional): logger object to use
    Returns:
        task arguments (to be passed to instantiate a Task Object)
    """

    def __init__(self, name, is_small=False, is_mini=False, longer_repeat=5, logger=None):
        logger = logger or logging.getLogger(__name__)

        name = name.lower()
        lookup_tables_dir_path = os.path.join(dir_path, name2dir[name])
        if not os.path.isdir(lookup_tables_dir_path):
            raise NotImplementedError(
                "Folder at {} does not exist".format(lookup_tables_dir_path))

        generation_arguments_path = os.path.join(
            lookup_tables_dir_path, 'generation_arguments.txt')
        if not os.path.isfile(generation_arguments_path):
            raise NotImplementedError(
                "Generation Arguments .txt Missing in Table Lookup Folder \
                - Cannot Generate Table")

        lookup_tables_data_dir_path = os.path.join(
            lookup_tables_dir_path, "data")

        if not os.path.isdir(lookup_tables_data_dir_path):
            logger.info(
                "Data not present for {} \n Generating Dataset".format(name))
            make_long_lookup_tables(
                lookup_tables_data_dir_path, generation_arguments_path)

        # Get default params from json
        # - these are not required but offer recommendation on default params
        default_params = get_default_params(lookup_tables_dir_path)

        # Update the defauls params if task is small /mini
        if default_params is not None:
            if is_small:
                default_params["task_defaults"]["k"] = 1
            if is_mini:
                default_params["task_defaults"]["k"] = 1
                default_params["task_defaults"]["batch_size"] = 128
                default_params["task_defaults"]["patience"] = 2
                default_params["task_defaults"]["epochs"] = 3
                default_params["task_defaults"]["n_attn_plots"] = 1

        train_file = "train"
        valid_file = "validation"
        test_files = flatten(["heldout_inputs", "heldout_compositions",
                              "heldout_tables",
                              "new_compositions", repeat(
                                  "longer_seen", longer_repeat),
                              repeat("longer_incremental", longer_repeat),
                              repeat("longer_new", longer_repeat)])

        super().__init__(name, lookup_tables_data_dir_path,
                         train_file, valid_file, test_files, default_params)
