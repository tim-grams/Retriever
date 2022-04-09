from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from src.preprocessing.preprocessing import tokenization
import nltk
from textblob import TextBlob


def cosine_similarity_score(embedding_1, embedding_2):
    return cosine_similarity(embedding_1, embedding_2)[0][0]


def euclidean_distance_score(embedding_1, embedding_2):
    return euclidean_distances(embedding_1, embedding_2)[0][0]


def manhattan_distance_score(embedding_1, embedding_2):
    return manhattan_distances(embedding_1, embedding_2)[0][0]


def jaccard(token_vector_1, token_vector_2):
    intersect = set(token_vector_1).intersection(set(token_vector_2))
    union = set(token_vector_1).union(set(token_vector_2))
    try:
        return len(intersect) / len(union)
    except ZeroDivisionError:
        return 0


def characters(sentence: str):
    return len(sentence.replace(' ', ''))


def words(preprocessed_tokens: list):
    return len(preprocessed_tokens)


def POS(sentence):
    tokens = list(tokenization(sentence))
    tags = nltk.pos_tag(tokens)
    nouns = len([tag[0] for tag in tags if tag[1].startswith('NN')])
    adj = len([tag[0] for tag in tags if tag[1].startswith('JJ')])
    verbs = len([tag[0] for tag in tags if tag[1].startswith('VB')])
    return nouns, adj, verbs


def subjectivity(sentence):
    return TextBlob(sentence).sentiment.subjectivity


def polarisation(sentence):
    return TextBlob(sentence).sentiment.polarity


def named_entities(sentence):
    pass


def stopwords(sentence):
    pass
