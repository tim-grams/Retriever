import tarfile
import requests
import gzip
import shutil
from pathlib import Path
from tqdm import tqdm


#ToDo: Replace remote_url with custom args to download specific (one or multiple) files or all files
# But rn still a bit unclear what we need so i didnt implement it yet
def download_dataset(remote_url):

    #Construct paths
    file_name = remote_url.rsplit("/", 1)[-1]
    proj_path = Path(__file__).resolve().parents[2]
    data_path = Path.joinpath(proj_path, "data")
    file_path =  Path.joinpath(data_path, file_name)

    #Get Data and Save on disk (streaming bc large filesizes, so we don't run out of memory)
    print("Start Downloading Data")
    response = requests.get(remote_url, stream=True)
    total_bytesize = int(response.headers.get('content-length', 0))
    block_size = 8192
    progress_bar = tqdm(total=total_bytesize, unit='iB', unit_scale=True)

    with open(file_path, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    print("Downloading finished")

    #Check if everything went right with the download
    if total_bytesize != 0 and progress_bar.n != total_bytesize:
        print("ERROR! ERROR! Something went wrong while downloading")
        return


    #unzip archives if needed
    if (file_path.name.endswith(".tar.gz")):
        print("start unzipping .tar.gz file")
        with tarfile.open(file_path) as tar:
            tar.extractall(path=data_path)
        print("unzipping successful")

    elif (file_path.name.endswith(".gz")):
        print("start unzipping .gz file")
        with gzip.open(file_path, "rb") as f_in:         
            with open(Path.joinpath(data_path, file_name[:-3]), "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        print("unzipping successful")        

