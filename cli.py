import logging
import argparse
from src.pipeline import Pipeline

LOGGER = logging.getLogger('cli')
pipeline = Pipeline()


def _setup(args):
    pipeline.setup()
    pipeline.preprocess()
    pipeline.create_train_features()
    pipeline.create_val_features()
    pipeline.create_test_features()


def _evaluate(args):
    pipeline.evaluate(**args)


def _get_parser():
    """ Sets up a command line interface.

    Args:
        parser (ArgumentParser): Argument parser.
    """
    logging_args = argparse.ArgumentParser(add_help=False)
    logging_args.add_argument('-v', '--verbose', action='count', default=0)
    logging_args.add_argument('-l', '--logfile')
    parser = argparse.ArgumentParser(description='Retrieval Command Line Interface',
                                     parents=[logging_args])
    subparsers = parser.add_subparsers(title='action', help='Action to perform')

    # Parser to setup, preprocess and create features
    data_download = subparsers.add_parser('setup', help='Setup, preprocess and create features')
    data_download.set_defaults(action=_setup)

    # Parser to create tfidf features
    data_download = subparsers.add_parser('evaluate', help='Evaluate point- and pairwise algorithms')
    data_download.add_argument('--model', help='Model used. Options are nb, lr, mlp', type=str, default='nb')
    data_download.add_argument('--pca', help='Decompose features into PCA components', type=int, default=0)
    data_download.add_argument('--pairwise', help='Pairwise model. Options are ranknet', type=str, default=None)
    data_download.set_defaults(action=_evaluate)

    return parser


def _logging_setup(verbosity: int = 1, logfile: str = None):
    """ Sets up a logging interface.
    Args:
        verbosity (int): Verbosity level of the logging output.
        logfile (str): Location of a logfile in which the logs can be saved.
    """
    logger = logging.getLogger()
    log_level = (2 - verbosity) * 10
    fmt = '%(asctime)s - %(levelname)s - %(module)s - %(message)s'
    formatter = logging.Formatter(fmt)
    logger.setLevel(log_level)
    logger.propagate = False

    if logfile:
        file_handler = logging.FileHandler(logfile)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logging.getLogger('retriever:worker').setLevel(logging.INFO)


def main():
    parser = _get_parser()
    args = parser.parse_args()

    _logging_setup(args.verbose, args.logfile)

    if not hasattr(args, 'action'):
        parser.print_help()
        parser.exit()

    args.action(args)


if __name__ == '__main__':
    main()