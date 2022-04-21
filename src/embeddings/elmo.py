from flair.embeddings import ELMoEmbeddings
from flair.data import Sentence
from nltk.tokenize.treebank import TreebankWordDetokenizer
from tqdm import tqdm
import numpy as np
import torch


class Elmo(object):
    def __init__(self):
        # init embedding
        self.embedding = ELMoEmbeddings()

    def transform(self, X):

        elmo_list = []
        for line in tqdm(X):
            detokenized = TreebankWordDetokenizer().detokenize(line)
            sentence = Sentence(detokenized)
            self.embedding.embed(sentence)
            input = torch.empty(sentence[0].embedding.size())
            tokens_per_sentence = torch.zeros_like(input)
            for token in sentence:
                tokens_per_sentence = torch.add(tokens_per_sentence, token.embedding)
            elmo_list.append(tokens_per_sentence)
        return np.array(elmo_list)