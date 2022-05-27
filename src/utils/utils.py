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


def check_file_exits(path: str = None):
    """ Checks if a file exists.

    Args:
        path (str): The path to check.

    Returns:
        (Boolean): True or False

    """
    if path is not None:
        return os.path.isfile(path)
    return False
    

def save(obj: object, path: str):
    """ Saves an object to a specified path.

    Args:
        obj (object): Object to save
        path (str): The path to store the object to.

    Returns:
        path (str): The path the object has been stores to

    """
    with open(path, 'wb') as fin:
        pickle.dump(obj, fin)
    return path


def load(path: str):
    """ Loads an object from a specified path.

    Args:
        path (str): The path to load the object from.

    Returns:
        (object): Returns the loaded object

    """
    return pickle.load(open(path, "rb"))
