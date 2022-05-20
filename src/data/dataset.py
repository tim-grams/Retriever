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
    ''' Combines and executes download and unzip methods
    
    Args:
        datasets (list): List of required files
        path (str): Location to store downloaded data.

    Returns:
        none
    '''

    links = {
        'collection.tsv': "https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz",
        'queries.train.tsv': "https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz",
        'qrels.train.tsv': "https://msmarco.blob.core.windows.net/msmarcoranking/qrels.train.tsv",
        'qidpidtriples.train.full.2.tsv': 'https://msmarco.blob.core.windows.net/msmarcoranking/qidpidtriples.train.full.2.tsv.gz',
        'msmarco-test2019-queries.tsv': 'https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz',
        'msmarco-test2020-queries.tsv': 'https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-test2020-queries.tsv.gz',
        '2019qrels-pass.txt': 'https://trec.nist.gov/data/deep/2019qrels-pass.txt',
        '2020qrels-pass.txt': 'https://trec.nist.gov/data/deep/2020qrels-pass.txt'
    }

    zip_links = {
        'collection.tsv': 'collection.tar.gz',
        'queries.train.tsv': 'queries.tar.gz',
        'qrels.train.tsv': 'qrels.train.tsv',
        'qidpidtriples.train.full.2.tsv': 'qidpidtriples.train.full.2.tsv.gz',
        'msmarco-test2019-queries.tsv': 'msmarco-test2019-queries.tsv.gz',
        'msmarco-test2020-queries.tsv': 'msmarco-test2020-queries.tsv.gz',
        '2019qrels-pass.txt': '2019qrels-pass.txt',
        '2020qrels-pass.txt': '2020qrels-pass.txt'
    }

    check_path_exists(path)

    for dataset in datasets:
        filepath = os.path.join(path, dataset)
        zippath = os.path.join(path, zip_links[dataset])

        if (not os.path.exists(zippath) and not os.path.exists(filepath)):
            download(links[dataset], path)
        else:
            LOGGER.debug(f'{dataset} archive already exists')

        unzip(os.path.join(path, zip_links[dataset])) if not os.path.exists(filepath) else LOGGER.debug(
            f'{dataset} already exists')


def download(remote_url: str = None, path: str = None):
    ''' Downloads files
    
    Args:
        remote_url (str): URL to dataset
        path (str): Location to store downloaded data.

    Returns:
        none

    '''
    assert remote_url is not None, "No URL given"
    assert path is not None, "Specify local path"

    file_name = remote_url.rsplit("/", 1)[-1]
    file_path = os.path.join(path, file_name)

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
    ''' Unzips files
    
    Args:
        file (str): Specify file to unzip
        
    Returns:
        none

    '''
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
            with open(file[:-3], "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        os.remove(file)
        LOGGER.info("unzipping successful")


def import_val_test_queries(path: str = "data/TREC_Passage", qrels_val: list = None, qrels_test: list = None):
    ''' Imports validation and test queries
    
    Args:
        path (str): Location of dataset
        qrels_val (list): 
        qrels_test (list): 

    Returns:
        val_df (pd.DataFrame): Query validation IDs and content 
        test_df (pd.DataFrame): Query test IDs and content

    '''
    filepath = os.path.join(path, 'msmarco-test2019-queries.tsv')
    if not os.path.exists(filepath):
        LOGGER.debug("File not there, downloading a new one")
        download_dataset(["msmarco-test2019-queries.tsv"], path)

    col_names = ["qID", "Query"]
    val_df = pd.read_csv(filepath, sep="\t", names=col_names, header=None)
    if qrels_val is not None:
        val_df = val_df[val_df['qID'].isin(qrels_val)].reset_index(drop=True)

    filepath = os.path.join(path, 'msmarco-test2020-queries.tsv')
    if not os.path.exists(filepath):
        LOGGER.debug("File not there, downloading a new one")
        download_dataset(["msmarco-test2019-queries.tsv"], path)

    col_names = ["qID", "Query"]
    test_df = pd.read_csv(filepath, sep="\t", names=col_names, header=None)
    if qrels_test is not None:
        test_df = test_df[test_df['qID'].isin(qrels_test)].reset_index(drop=True)
    return val_df, test_df


def import_queries(path: str = "data/TREC_Passage", collection: list = None):
    ''' Imports train queries
    
    Args:
        path (str): Location of dataset
        collection (list): 
        qrels_test (list): 

    Returns:
        df (pd.DataFrame): Query train IDs and content 

    '''
    filepath = os.path.join(path, 'queries.train.tsv')
    if not os.path.exists(filepath):
        LOGGER.debug("File not there, downloading a new one")
        download_dataset(["queries.train.tsv"], path)

    col_names = ["qID", "Query"]
    df = pd.read_csv(filepath, sep="\t", names=col_names, header=None)
    if collection is not None:
        df = df[df['qID'].isin(collection)].reset_index(drop=True)

    return df


def import_collection(path: str = "data/TREC_Passage", qrels_val: list = None, qrels_test: list = None, triples: list = None, samples: int = 0):
    ''' Imports data from collection.tsv file
    
    Args:
        path (str): Location of dataset
        qrels_val (list):
        triples (list): 
        samples (int): Specify number of rows to be imported from dataset

    Returns:
        df (pd.DataFrame): Data frame containing IDs and Passages from collection dataset

    '''
    filepath = os.path.join(path, 'collection.tsv')
    if not os.path.exists(filepath):
        LOGGER.debug("File not there, downloading a new one")
        download_dataset(['collection.tsv'], path)

    col_names = ["pID", "Passage"]
    df = pd.read_csv(filepath, sep="\t", names=col_names, header=None)
    if samples > 0:
        sampling = df.sample(samples, random_state=42).reset_index(drop=True)
    if qrels_val is not None and qrels_test is not None and triples is not None:
        df = df[(df['pID'].isin(qrels_val)) | (df['pID'].isin(qrels_test)) | (df['pID'].isin(triples))].reset_index(drop=True)
    if samples > 0:
        df = pd.concat([sampling, df]).reset_index(drop=True)
    return df


def import_qrels(path: str = "data/TREC_Passage", samples: int = 5):
    ''' Imports data from 2019qrels-pass.txt as validation set and from 2020qrels-pass.txt as test set
    
    Args:
        path (str): Location of dataset
        samples (int): Specify number of rows to be imported from dataset        
        
    Returns:
        df_val (pd.DataFrame): Data frame containing validation set
        df_test (pd.DataFrame): Data frame containing test set 

    '''
    filepath = os.path.join(path, '2019qrels-pass.txt')
    if not os.path.exists(filepath):
        LOGGER.debug("File not there, downloading a new one")
        download_dataset(['2019qrels-pass.txt'], path)

    col_names = ["qID", "0", "pID", "feedback"]
    df_val = pd.read_csv(filepath, sep=" ", names=col_names, header=None)
    df_val = df_val[df_val['feedback'] >= 1]
    sampled_qids = pd.Series(df_val['qID'].unique()).sample(samples, random_state=42).reset_index(drop=True)
    df_val = df_val[df_val['qID'].isin(sampled_qids)].reset_index(drop=True)

    filepath = os.path.join(path, '2020qrels-pass.txt')
    if not os.path.exists(filepath):
        LOGGER.debug("File not there, downloading a new one")
        download_dataset(['2020qrels-pass.txt'], path)

    col_names = ["qID", "0", "pID", "feedback"]
    df_test = pd.read_csv(filepath, sep=" ", names=col_names, header=None)
    df_test = df_test[df_test['feedback'] >= 1]
    sampled_qids = pd.Series(df_test['qID'].unique()).sample(samples, random_state=42).reset_index(drop=True)
    df_test = df_test[df_test['qID'].isin(sampled_qids)].reset_index(drop=True)
    return df_val.drop(columns=['0']), df_test.drop(columns=['0'])


def import_training_set(path: str = "data/TREC_Passage", samples: int = 200):
    ''' Imports data from qidpidtriples.train.full.2.tsv as training set
    
    Args:
        path (str): Location of dataset
        samples (int): Specify number of rows to be imported from dataset          
        
    Returns:
        df (pd.DataFrame): Data frame containing training set

    '''    
    filepath = os.path.join(path, 'qidpidtriples.train.full.2.tsv')
    if not os.path.exists(filepath):
        LOGGER.debug("File not there, downloading a new one")
        download_dataset(['qidpidtriples.train.full.2.tsv'], path)

    col_names = ["qID", "positive", "negative"]
    df = pd.read_csv(filepath, sep="\t", names=col_names, header=None)
    df = df.sample(samples, random_state=42).reset_index(drop=True)
    return pd.DataFrame({
        'qID': pd.concat([df['qID'], df['qID']]),
        'pID': pd.concat([df['positive'], df['negative']]),
        'y': [1] * len(df) + [0] * len(df)
    }).drop_duplicates()
