from flair.embeddings import ELMoEmbeddings
from flair.data import Sentence
from tqdm import tqdm
import numpy as np
from nltk.tokenize.treebank import TreebankWordDetokenizer


class Elmo(object):
    def __init__(self):
        # init embedding
        self.embedding = ELMoEmbeddings()

    def fit_transform(self, X):

        elmo_list = []
        for line in tqdm(X):
            dict_emb = {}
            detokenized = TreebankWordDetokenizer().detokenize(line)
            sentence = Sentence(detokenized)
            self.embedding.embed(sentence)
            for token in sentence:
                dict_emb[token] = token.embedding
            elmo_list.append(dict_emb)
        return np.array(elmo_list)
    