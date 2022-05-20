from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from src.data.preprocessing import tokenization
import nltk
from textblob import TextBlob
import numpy as np


def cosine_similarity_score(embedding_1: list, embedding_2: list):
    ''' Calculates cosine similarity between two word embeddings.

    Args:
        embedding_1 (list): List contaning word embedding
        embedding_2 (list): List contaning word embedding

    Returns:
        cosine_similarity (float): Cosine similarity between two word embeddings

    ''' 
    return cosine_similarity(embedding_1, embedding_2)[0][0]


def euclidean_distance_score(embedding_1: list, embedding_2: list):
    ''' Calculates euclidean distance between two word embeddings.

    Args:
        embedding_1 (list): List contaning word embedding
        embedding_2 (list): List contaning word embedding

    Returns:
        euclidean_distances (float): Euclidean distance between two word embeddings

    ''' 
    return euclidean_distances(embedding_1, embedding_2)[0][0]


def manhattan_distance_score(embedding_1: list, embedding_2: list):
    ''' Calculates manhatten distance between two word embeddings.

    Args:
        embedding_1 (list): List contaning word embedding
        embedding_2 (list): List contaning word embedding

    Returns:
        manhattan_distances (float): Manhatten distance between two word embeddings

    ''' 
    return manhattan_distances(embedding_1, embedding_2)[0][0]


def jaccard(token_vector_1: list, token_vector_2: list):
    ''' Calculates jaccard coefficient between two lists of tokens

    Args:
        token_vector_1 (list): List contaning tokens
        token_vector_2 (list): List contaning tokens

    Returns:
        (float): Jaccard coefficient as float

    ''' 
    intersect = set(token_vector_1).intersection(set(token_vector_2))
    union = set(token_vector_1).union(set(token_vector_2))
    try:
        return len(intersect) / len(union)
    except ZeroDivisionError:
        return 0


def characters(sentence: str):
    ''' Returns length of sentence, not considering whitespaces

    Args:
        sentence (str): Sentence as string

    Returns:
        (int): Length of sentence as int

    ''' 
    return len(sentence.replace(' ', ''))


def words(sentence: str):
    ''' Returns number of words in a sentence

    Args:
        sentence (str): Sentence as string

    Returns:
        (int): Number of words in a sentence as int

    '''
    tokens = list(tokenization(sentence))
    return len(tokens)


def POS(sentence: str):
    ''' Returns Number of nouns, adjectives and verbs in a sentence

    Args:
        sentence (str): Sentence as string

    Returns:
        nouns (int): Number of nouns
        adj (int): Number of adjectives
        vetbs (int): Number of verbs

    '''
    tokens = list(tokenization(sentence))
    tags = nltk.pos_tag(tokens)
    nouns = len([tag[0] for tag in tags if tag[1].startswith('NN')])
    adj = len([tag[0] for tag in tags if tag[1].startswith('JJ')])
    verbs = len([tag[0] for tag in tags if tag[1].startswith('VB')])
    return nouns, adj, verbs


def subjectivity(sentence: str):
    ''' Returns the subjectivity of a sentence

    Args:
        sentence (str): Sentence as string

    Returns:
        subjectivity (int): float within the range [0.0, 1.0] where 0.0 is very objective and 1.0 is very subjective

    '''
    return TextBlob(sentence).sentiment.subjectivity


def polarisation(sentence :str):
    ''' Returns the polarisation of a sentence

    Args:
        sentence (str): Sentence as string

    Returns:
        polarisation (int): float within the range [-1.0, 1.0]

    '''
    return TextBlob(sentence).sentiment.polarity


def difference(doc_count, query_count):
    ''' Returns the absolut difference between doc_count and query_count

    Args:
        doc_count (int):
        query_count (int): 

    Returns:
        (int): Absolut difference between doc_count and query_count

    '''
    return np.abs(doc_count - query_count)


def relative_difference(doc_count, query_count):
    ''' Returns the relative difference between doc_count and query_count

    Args:
        doc_count (int):
        query_count (int): 

    Returns:
        (float): Relative difference between doc_count and query_count

    '''
    return doc_count / query_count
