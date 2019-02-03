import yaml
import os
import warnings


def get_default_params(path):
    """
    Reads the default_params yml file in the directory passed in path
    Args: path to folder where .yml file is contained
    Returns: - None if .yml file does not exist
             - Or a dictionary object of the json if it does
    Raises: YAMLError if yaml package unable to read file
    """

    path = os.path.join(path, "default_params.yml")

    default_params = None
    if not os.path.isfile(path):
        warnings.warn("Default Params File Missing at {} \n \
        but still invoked default_params function".format(path), Warning)
    else:
        with open(path, 'r') as stream:
            try:
                default_params = yaml.load(stream)
            except yaml.YAMLError as exc:
                raise exc

    return default_params
