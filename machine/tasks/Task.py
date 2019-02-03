import os


class Task(object):
    """Helper class containing meta information of datasets.

    Args:
        name (str): name of the dataset.
        data_dir (str):  directory to prepend to all path above.
        train_path (str): path to training data.
        test_paths (list of str): list of paths to all test data.
        valid_path (str): path to validation data.
        default_params (None or Dict): default params that represent baseline
        extension (str, optional): extension to add to every paths above.
    """

    def __init__(self,
                 name,
                 data_dir,
                 train_filename,
                 valid_filename,
                 test_filenames,
                 default_params,
                 extension="tsv"):

        self.name = name
        self.extension = "." + extension
        self.data_dir = data_dir
        self.train_path = self._add_presufixes(train_filename)
        self.valid_path = self._add_presufixes(valid_filename)
        self.test_paths = [self._add_presufixes(
            path) for path in test_filenames]

        self.default_params = default_params

        self._validate_all_filepaths()

    def _add_presufixes(self, path):
        if path is None:
            return None
        return os.path.join(self.data_dir, path) + self.extension

    def __repr__(self):
        return "{} Task".format(self.name)

    def _validate_all_filepaths(self):
        """
        Returns Error if a path is invalid in the stored paths
        """
        paths = [self.train_path, self.valid_path] + self.test_paths

        for p in paths:
            if not os.path.isfile(p):
                raise NameError("File at {} does not exist".format(p))
