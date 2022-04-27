from flair.embeddings import ELMoEmbeddings
from flair.data import Sentence
from nltk.tokenize.treebank import TreebankWordDetokenizer
from tqdm import tqdm
import torch
from src.utils.utils import check_path_exists, save, load
import os


class Elmo(object):
    def __init__(self):
        # init embedding
        self.embedding = ELMoEmbeddings()

    def transform(self, X: pd.Series, store: str = None):

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

        if store is not None:
            check_path_exists(os.path.dirname(store))
            save(elmo_list, store)
        return elmo_list