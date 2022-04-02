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


def download_dataset(datasets: str = "all", location: str = "data/TREC_Passage"):
    if datasets == "all":
        remote_url = "https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz"
    elif datasets == "queries":
        remote_url = "https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz"
    else:
        raise NotImplementedError

    check_path_exists(location)

    # Construct paths
    file_name = remote_url.rsplit("/", 1)[-1]
    file_path = os.path.join(location, file_name)

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

    # Check if everything went right with the download
    if total_bytesize != 0 and progress_bar.n != total_bytesize:
        LOGGER.error("Something went wrong while downloading")
        raise FileExistsError

    # unzip archives if needed
    if file_name.endswith(".tar.gz"):
        LOGGER.info("start unzipping .tar.gz file")
        with tarfile.open(file_path) as tar:
            tar.extractall(path=location)

        os.remove(file_path)
        LOGGER.info("unzipping successful")

    elif file_name.endswith(".gz"):
        LOGGER.info("start unzipping .gz file")
        with gzip.open(file_path, "rb") as f_in:
            with open(os.path.join(location, file_name[:-3]), "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        os.remove(file_path)
        LOGGER.info("unzipping successful")

    return location


def import_queries(filepath: str = "data/TREC_Passage/queries.dev.tsv",
                   dataframe_location: str = "data/raw"):
    if not os.path.exists(filepath):
        LOGGER.debug("File not there, downloading a new one")
        download_dataset("queries")

    check_path_exists(dataframe_location)
    dataframe_path = os.path.join(dataframe_location, 'queries.pkl')

    col_names = ["qID", "Query"]
    df = pd.read_csv(filepath, sep="\t", names=col_names, header=None)
    df.to_pickle(dataframe_path)
    return df


def import_collection(filepath: str = "data/TREC_Passage/collection.tsv",
                      dataframe_location: str = "data/raw"):
    if not os.path.exists(filepath):
        LOGGER.debug("File not there, downloading a new one")
        download_dataset()

    check_path_exists(dataframe_location)
    dataframe_path = os.path.join(dataframe_location, 'collection.pkl')

    col_names = ["pID", "Passage"]
    df = pd.read_csv(filepath, sep="\t", names=col_names, header=None)
    df.to_pickle(dataframe_path)
    return df
