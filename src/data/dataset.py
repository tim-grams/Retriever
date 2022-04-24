import tarfile
import requests
import gzip
import shutil
from tqdm import tqdm
import pandas as pd
import logging
import os
from src.utils.utils import check_path_exists

LOGGER = logging.getLogger('cli')


def download_dataset(datasets: list = None, path: str = "data/TREC_Passage"):
    assert datasets is not None, "No dataset selected"

    links = {
        'collection.tsv': "https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz",
        'queries.train.tsv': "https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz",
        'qrels.train.tsv': "https://msmarco.blob.core.windows.net/msmarcoranking/qrels.train.tsv",
        'qidpidtriples.train.full.2.tsv': 'https://msmarco.blob.core.windows.net/msmarcoranking/qidpidtriples.train.full.2.tsv',
        'msmarco-test2019-queries.tsv': 'https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv',
        '2019qrels-pass.txt': 'https://msmarco.blob.core.windows.net/msmarcoranking/2019qrels-pass.txt'
    }

    check_path_exists(path)

    for dataset in datasets:
        filepath = os.path.join(path, dataset)
        download(links[dataset], path) if not os.path.exists(filepath) else LOGGER.debug(f'{dataset} already exists')
        unzip(filepath)


def download(remote_url: str = None, path: str = None):
    assert remote_url is not None, "No URL given"
    assert path is not None, "Specify local path"

    # Construct paths
    file_name = remote_url.rsplit("/", 1)[-1]
    file_path = os.path.join(path, file_name)

    # Get Data and Save on disk (streaming bc large file sizes, so we don't run out of memory)
    LOGGER.info("Start Downloading Data")
    response = requests.get(remote_url, stream=True)
    total_bytesize = int(response.headers.get('content-length', 0))
    block_size = 8192
    progress_bar = tqdm(total=total_bytesize, unit='iB', unit_scale=True)

    with open(file_path, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    LOGGER.info("Downloading finished")

    if total_bytesize != 0 and progress_bar.n != total_bytesize:
        LOGGER.error("Something went wrong while downloading")
        raise FileExistsError


def unzip(file: str = None):
    assert file is not None, "No file specified"

    if file.endswith(".tar.gz"):
        LOGGER.info("start unzipping .tar.gz file")
        with tarfile.open(file) as tar:
            tar.extractall(path=os.path.dirname(file))

        os.remove(file)
        LOGGER.info("unzipping successful")

    elif file.endswith(".gz"):
        LOGGER.info("start unzipping .gz file")
        with gzip.open(file, "rb") as f_in:
            with open(os.path.join(os.path.dirname(file), file[:-3]), "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        os.remove(file)
        LOGGER.info("unzipping successful")


def import_queries(path: str = "data/TREC_Passage", queries: list = None):
    filepath = os.path.join(path, 'queries.train.tsv')
    if not os.path.exists(filepath):
        LOGGER.debug("File not there, downloading a new one")
        download_dataset(["queries.train.tsv"], path)

    col_names = ["qID", "Query"]
    df = pd.read_csv(filepath, sep="\t", names=col_names, header=None)
    if queries is not None:
        df = df[df['qID'].isin(queries)].reset_index(drop=True)

    filepath = os.path.join(path, 'msmarco-test2019-queries.tsv')
    if not os.path.exists(filepath):
        LOGGER.debug("File not there, downloading a new one")
        download_dataset(["msmarco-test2019-queries.tsv"], path)

    col_names = ["qID", "Query"]
    test_df = pd.read_csv(filepath, sep="\t", names=col_names, header=None)
    return df, test_df


def import_collection(path: str = "data/TREC_Passage", samples: int = 5000):
    filepath = os.path.join(path, 'collection.tsv')
    if not os.path.exists(filepath):
        LOGGER.debug("File not there, downloading a new one")
        download_dataset(['collection.tsv'], path)

    col_names = ["pID", "Passage"]
    df = pd.read_csv(filepath, sep="\t", names=col_names, header=None)
    df = df.sample(samples, random_state=42).reset_index(drop=True)
    return df


def import_qrels(path: str = "data/TREC_Passage"):
    filepath = os.path.join(path, '2019qrels-pass.txt')
    if not os.path.exists(filepath):
        LOGGER.debug("File not there, downloading a new one")
        download_dataset(['2019qrels-pass.txt'], path)

    col_names = ["qID", "0", "pID", "feedback"]
    df_test = pd.read_csv(filepath, sep=" ", names=col_names, header=None)
    return df_test.drop(columns=['0'])


def import_training_set(path: str = "data/TREC_Passage", collection: list = None):
    filepath = os.path.join(path, 'qidpidtriples.train.full.2.tsv')
    if not os.path.exists(filepath):
        LOGGER.debug("File not there, downloading a new one")
        download_dataset(['qidpidtriples.train.full.2.tsv'], path)

    col_names = ["qID", "positive", "negative"]
    df = pd.read_csv(filepath, sep="\t", names=col_names, header=None)
    if collection is not None:
        df = df[(df['positive'].isin(collection)) & (df['negative'].isin(collection))].reset_index(drop=True)
    return pd.DataFrame({
                'qID': pd.concat([df['qID'], df['qID']]),
                'pID': pd.concat([df['positive'], df['negative']]),
                'y': [1] * len(df) + [0] * len(df)
            }).drop_duplicates()
