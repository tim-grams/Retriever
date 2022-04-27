from flair.embeddings import WordEmbeddings
from flair.data import Sentence
import pandas as pd
import numpy as np
from src.utils.utils import check_path_exists, save, load
import os
from tqdm import tqdm
import torch
from nltk.tokenize.treebank import TreebankWordDetokenizer

class Glove(object):

    glove = None

    def __init__(self):
        self.glove = WordEmbeddings('glove')


    def transform(self, text_in_tokens: pd.Series, store: str = None):
        glove_vec = []

        for line in tqdm(text_in_tokens):
            detokenized = TreebankWordDetokenizer().detokenize(line)
            sentence = Sentence(detokenized)
            self.glove.embed(sentence)
            input = torch.empty(sentence[0].embedding.size())
            token_per_sentence = torch.zeros_like(input)
            #input = np.empty(sentence[0].embedding.size())
            #token_per_sentence = np.zeros(sentence[0].embedding.size())

            for token in sentence:
                token_per_sentence = torch.add(token_per_sentence, token.embedding)
            glove_vec.append(token_per_sentence.numpy())

        #glove_vec = np.array(glove_vec, dtype="object")
        
        if store is not None:
            check_path_exists(os.path.dirname(store))
            save(glove_vec, store)

        return glove_vec