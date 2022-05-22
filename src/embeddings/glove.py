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
    
    
"""   def transform_tfidfweighted(self, text_in_tokens: pd.Series, tf_idf_weights: pd.Series, store: str = None ):

        tfidf_glove_vec = [] # the tfidf-ft for each sentence/review is stored in this list
        
        for count, line in enumerate(tqdm(text_in_tokens)): # for each sentence
            detokenized = TreebankWordDetokenizer().detokenize(line)
            sentence = Sentence(detokenized)
            self.glove.embed(sentence)
            input = torch.empty(sentence[0].embedding.size())
            token_per_sentence = torch.zeros_like(input)
            tfidf_weights = tf_idf_weights.at[count]
            weight_sum = 0

            for count2 , token in enumerate(sentence): # for each word in a sentence
                    vec = token.embedding
                    raw_token = line[count2]
                    # obtain the tf_idfidf of a word in a sentence
                    tfidf = tfidf_weights[raw_token]
                    token_per_sentence = torch.add(token_per_sentence, (vec * tfidf))
                    weight_sum += tfidf
                    
                    token_per_sentence = torch.add(token_per_sentence, vec)

            torch.div(token_per_sentence, weight_sum)
            tfidf_glove_vec.append(token_per_sentence.numpy()) """
