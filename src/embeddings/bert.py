from msilib.schema import TextStyle
from sentence_transformers import SentenceTransformer, util
import numpy as np
import nltk
import pandas as pd
from tqdm import tqdm
from src.utils.utils import check_path_exists, save, load
import os



class Bert(object):

    def __init__(self):
        self.model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1") #Not the best model in terms of similarity scores but only a bit worse then the best one but only 20% the Size
        nltk.download("punkt")

    #Dont Preprocess Texts beforehand! 
    def transform(self, raw_texts: pd.Series, store: str = None):
        bert_dict = {}

        for text in tqdm(raw_texts):

            #Splits text into sentences and cuts off after the second sentence (Bert can only handle 1 or 2 sentences)
            sentences = nltk.tokenize.sent_tokenize(text) 
            if len(sentences) > 2:
                sentences = sentences[0:2]
            
            embedding = self.model.encode(sentences)
            bert_dict[text] = embedding
        
        if store is not None:
            check_path_exists(os.path.dirname(store))
            save(bert_dict, store)
            
        return np.array(bert_dict)





            
            

            



        
