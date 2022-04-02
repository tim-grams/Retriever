import tarfile
import requests
import gzip
import shutil
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import logging

LOGGER = logging.getLogger('cli')


def download_dataset(datasets="all"):
    if datasets == "all":
        remote_url = "https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz"
    elif datasets == "queries":
        remote_url = "https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz"
    else:
        raise NotImplementedError

    # Construct paths
    file_name = remote_url.rsplit("/", 1)[-1]
    proj_path = Path(__file__).resolve().parents[2]
    data_path = Path.joinpath(proj_path, "data")
    file_path = Path.joinpath(data_path, file_name)

    # Get Data and Save on disk (streaming bc large filesizes, so we don't run out of memory)
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
        LOGGER.error("ERROR! ERROR! Something went wrong while downloading")
        return

    # unzip archives if needed
    if file_path.name.endswith(".tar.gz"):
        LOGGER.info("start unzipping .tar.gz file")
        with tarfile.open(file_path) as tar:
            tar.extractall(path=data_path)
        LOGGER.info("unzipping successful")

    elif file_path.name.endswith(".gz"):
        LOGGER.info("start unzipping .gz file")
        with gzip.open(file_path, "rb") as f_in:
            with open(Path.joinpath(data_path, file_name[:-3]), "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        LOGGER.info("unzipping successful")


def import_queries(filepath=""):
    if filepath == "":
        proj_path = Path(__file__).resolve().parents[2]
        filepath = Path.joinpath(proj_path, "data\queries.dev.tsv")
        if not Path.is_file(filepath):
            download_dataset("queries")

    colnames = ["qID", "Query"]
    df = pd.read_csv(filepath, sep="\t", names=colnames, header=None)
    return df


def import_collection(filepath=""):
    if filepath == "":
        proj_path = Path(__file__).resolve().parents[2]
        filepath = Path.joinpath(proj_path, "data\queries.dev.tsv")
        if not Path.is_file(filepath):
            LOGGER.debug("File not there, downloading a new one")
            download_dataset()

    colnames = ["pID", "Passage"]
    df = pd.read_csv(filepath, sep="\t", names=colnames, header=None)
    return df


"""def import_queries(filepath, linecount):

    #size = Path(filepath).stat().st_size
    #print(size)
    #targetsize = size*(outputpercent/100)     
    #print(targetsize)

    datalist = list() # List of tuple(index, query)
    querylist = list() # list of only queries as strings
    with open(filepath) as tsvfile:
        csvreader = csv.reader(tsvfile, delimiter="\t")
        for line in csvreader:        
            #querylist.append(line[1])
            datalist.append((int(line[0]), line[1]))
            if len(datalist) >= linecount or len(querylist) >= linecount: 
                break


    #query_df = pd.DataFrame (querylist, columns=["Passage"])
    query_df = pd.DataFrame (datalist, columns=[ "Pid" , "Passage" ])
    return query_df"""
