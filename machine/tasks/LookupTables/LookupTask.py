from machine.tasks import Task
from machine.tasks.utils import get_default_params, download_file_from_google_drive, unzip_and_remove_zip

import os

dir_path = os.path.dirname(os.path.realpath(__file__))


class LookupTask(Task):
    """
    The lookup table dataset as as Task Object

    Args:
        is_small (bool, optional) - Whether to make default params have lower default k
        is_mini (bool, optional) - Whether to make default params be optimized for training speed
                                 - k=1;batch_size=128;patience=2;epochs=3

    Raises: 
        NotImplementedError: In case the data folder is missing and must be downloaded

    Note: this is a subclass of the Task object and initializes its super class (Task)
          with the following parameters:
            - "lookup" (task name)
            - lookup_tables_data_dir_path, (directory of lookup table data files)
            - train_file (str) - name of training file
            - valid_file (str) - name of validation file
            - test_files (str) - name of test file
            - default_params (None or dictionary if present) - offers training suggestions
    """

    def __init__(self, is_small=False, is_mini=False):

        lookup_tables_data_dir_path = os.path.join(dir_path, "data")

        if not os.path.isdir(lookup_tables_data_dir_path):
            self._download_lookup_tables()

        data_dir = os.path.join(
            lookup_tables_data_dir_path, "samples/sample1/")

        test_files = ["heldout_inputs", "heldout_compositions", "heldout_tables",
                      "new_compositions", "longer_compositions_seen",
                      "longer_compositions_incremental", "longer_compositions_new"]

        valid_file = "validation"
        train_file = "train"

        # Get default params from yml file
        # - these are not required but offer recommendation on default params
        default_params = get_default_params(dir_path)

        # Update the defauls params if task is small /mini
        if is_small:
            default_params["task_defaults"]["k"] = 1
        if is_mini:
            default_params["task_defaults"]["k"] = 1
            default_params["task_defaults"]["batch_size"] = 128
            default_params["task_defaults"]["patience"] = 2
            default_params["task_defaults"]["epochs"] = 3
            default_params["task_defaults"]["n_attn_plots"] = 1

        super().__init__("lookup", data_dir,
                         train_file, valid_file, test_files, default_params)

    def _download_lookup_tables(self):
        """
        Downloads the lookup-3bit zip from a url and unzips/untar it into LookupTables/data
        """
        # url = "https://drive.google.com/file/d/1DyWeYjUXlwW4mwBaicEHzyasW1DkRIhZ/view?usp=sharing"
        file_id = '1DyWeYjUXlwW4mwBaicEHzyasW1DkRIhZ'
        destination_file = os.path.join(dir_path, "data.zip")
        download_file_from_google_drive(file_id, destination_file)
        unzip_and_remove_zip(dir_path, destination_file)
