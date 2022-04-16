from flair.embeddings import WordEmbeddings
from flair.data import Sentence
import pandas as pd
import numpy as np
from src.utils.utils import check_path_exists, save, load
import os
from tqdm import tqdm
import h5py
from nltk.tokenize.treebank import TreebankWordDetokenizer

class Glove(object):

    glove = None

    def __init__(self):
        self.glove = WordEmbeddings('glove')


    def transform(self, text_in_tokens: pd.Series, store: str = None):
        glove_vec = []

        for line in tqdm(text_in_tokens):
            glove_dict = {}
            detokenized = TreebankWordDetokenizer().detokenize(line)
            sentence = Sentence(detokenized)
            self.glove.embed(sentence)

            for token in sentence:
                glove_dict[token] = token.embedding
            glove_vec.append(glove_dict)

        return np.array(glove_vec)