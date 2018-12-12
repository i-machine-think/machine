from machine.tasks import Task
from machine.tasks.utils import get_default_params, download_file_from_google_drive, unzip_and_remove_zip
import os

dir_path = os.path.dirname(os.path.realpath(__file__))


class SymbolTask(Task):
    """
    Gets the symbol rewriting data in Task Object format
    Args:
        is_small (bool, optional) - Whether to make default params have lower default k 
        is_mini (bool, optional) - Whether to make default params be optimized for training speed
                                 - k=1;batch_size=128;patience=2;epochs=3

    Raises: 
        NotImplementedError: In case the data folder is missing and must be downloaded

    Note: this is a subclass of the Task object and initializes its super class (Task)
          with the following parameters:
            - 'Symbol Rewriting' (task name)
            - data_dir, (directory of symbol rewriting/data files)
            - train_file (str) - name of training file
            - valid_file (str) - name of validation file
            - test_files (str) - name of test file
            - default_params (must be present as json in SymbolRewriting folder) 
                - offers training suggestions
    """

    def __init__(self, is_small=False, is_mini=False):

        data_dir = os.path.join(dir_path, "data")

        if not os.path.isdir(data_dir):
            self._download_symbol_rewriting_data()

        train_file = "grammar_std.train.full"
        test_files = ["grammar_long.tst.full", "grammar_repeat.tst.full",
                      "grammar_short.tst.full", "grammar_std.tst.full"]
        valid_file = "grammar.val"

        # Get default params from yml
        # - these are not required but offer recommendation on default params
        default_params = get_default_params(dir_path)

        if is_small:
            train_file = "grammar_std.train.small"
            default_params["task_defaults"]["k"] = 1
        if is_mini:
            train_file = "grammar_std.train.small"
            default_params["task_defaults"]["k"] = 1
            default_params["task_defaults"]["batch_size"] = 128
            default_params["task_defaults"]["patience"] = 2
            default_params["task_defaults"]["epochs"] = 3
            default_params["task_defaults"]["n_attn_plots"] = 1

        super().__init__("Symbol Rewriting", data_dir, train_file,
                         valid_file, test_files, default_params)

    def _download_symbol_rewriting_data(self):
        """
        Downloads the symbol rewriting zip from a url and unzips/untar it into current directory as data folder

        Args: 
            data_dir_path (path): where to download and unzip the lookup-3bit data
        """
        # url = "https://drive.google.com/file/d/1XbVZmH0-w8h1VHUYZvXWLV8E4smQPQyz/view?usp=sharing"
        file_id = '1XbVZmH0-w8h1VHUYZvXWLV8E4smQPQyz'
        destination_file = os.path.join(dir_path, "data.zip")
        download_file_from_google_drive(file_id, destination_file)
        unzip_and_remove_zip(dir_path, destination_file)
