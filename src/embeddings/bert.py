from sentence_transformers import SentenceTransformer
import pandas as pd
from tqdm import tqdm
from src.utils.utils import check_path_exists, save, load
import os


class Bert(object):
    ''' A class to create bert embeddings.

    Methods:
    transform(raw_texts: pd.Series, store: str = None):
        Transforms series of unpreprocessed strings to bert embeddings
    ''' 
    def __init__(self):
        ''' Constructs bert object using a pretrained model. '''       
        self.model = SentenceTransformer(
            "multi-qa-MiniLM-L6-cos-v1")

    # Dont Preprocess Texts beforehand!
    def transform(self, raw_texts: pd.Series, store: str = None):
        ''' Transform Series of unpreprocessed strings to bert embeddings.
    
        Args:
            raw_texts (pd.Series): Series of unpreprocessed strings

        Returns:
            bert_vec (list): List containing bert embeddings

        '''   
        bert_vec = []

        for text in tqdm(raw_texts):
            embedding = self.model.encode(text)
            bert_vec.append(embedding)

        if store is not None:
            check_path_exists(os.path.dirname(store))
            save(bert_vec, store)

        return bert_vec
