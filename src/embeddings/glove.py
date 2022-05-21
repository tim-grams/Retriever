from flair.embeddings import WordEmbeddings
from flair.data import Sentence
import pandas as pd
from src.utils.utils import check_path_exists, save, load
import os
from tqdm import tqdm
import torch
from nltk.tokenize.treebank import TreebankWordDetokenizer


class Glove(object):
    ''' A class to create glove embeddings.

    Methods:
    transform(text_in_tokens: pd.Series, store: str = None)
        Transform series of preprocessed tokens to glove embeddings
    '''
    glove = None

    def __init__(self):
        ''' Constructs glove object using a pretrained model. ''' 
        self.glove = WordEmbeddings('glove')

    def transform(self, text_in_tokens: pd.Series, store: str = None):
        ''' Transform series of preprocessed tokens to glove embeddings.
    
        Args:
            text_in_tokens (pd.Series): Series of preprocessed tokens

        Returns:
            glove_vec (list): List containing glove embeddings

        '''      
        glove_vec = []

        for line in tqdm(text_in_tokens):
            detokenized = TreebankWordDetokenizer().detokenize(line)
            sentence = Sentence(detokenized)
            self.glove.embed(sentence)
            input = torch.empty(sentence[0].embedding.size())
            token_per_sentence = torch.zeros_like(input)

            for token in sentence:
                token_per_sentence = torch.add(token_per_sentence, token.embedding)
            glove_vec.append(token_per_sentence.numpy())

        if store is not None:
            check_path_exists(os.path.dirname(store))
            save(glove_vec, store)

        return glove_vec
