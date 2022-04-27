import os
import logging
import dill as pickle

LOGGER = logging.getLogger('utils')


def check_path_exists(path: str = None):
    """ Checks if a path exists and creates it if not.
    Args:
        path (str): The path to check.
    """
    if path is not None and not os.path.exists(path):
        LOGGER.info(f'Creating path {path}')
        os.makedirs(path)


def save(obj: object, path: str):
    with open(path, 'wb') as fin:
        pickle.dump(obj, fin)
    return path


def load(path: str):
    return pickle.load(open(path, "rb"))