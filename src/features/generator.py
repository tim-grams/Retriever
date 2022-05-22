import pandas as pd
from tqdm import tqdm
import logging
import numpy as np
import nltk
from src.embeddings.bert import Bert
from src.embeddings.tfidf import TFIDF
from src.embeddings.glove import Glove
from src.embeddings.word2vec import word2vec
from src.features.features import cosine_similarity_score, euclidean_distance_score, manhattan_distance_score, jaccard, \
    words, relative_difference, characters, difference, subjectivity, polarisation, POS
from src.utils.utils import load
from src.features.bm25 import BM25

nltk.download('averaged_perceptron_tagger')

tqdm.pandas()
LOGGER = logging.getLogger('generator')


def create_all(features: pd.DataFrame, collection: pd.DataFrame, queries: pd.DataFrame, tfidf=None, glove=None, bert=None, w2v=None):
    ''' Creates all implemented embeddings (bert, glove, tfidf, word2vec) 
        and features (cosine, euclidean, manhattan, jaccard, sentence, interpretation, BM25, POS).

    Args:
        features (pd.DataFrame): Dataframe containing feature data
        collection (pd.DataFrame):  Dataframe containing collection data
        queries (pd.DataFrame): Dataframe containing query data
        tfidf (TFIDF object): Creates new object of class tfidf if None
        glove (Glove object): Creates new object of class Glove if None
        bert (Bert object): Creates new object of class Bert if None
        w2v (word2vec object): Creates new object of class word2vec if None

    Returns:
        features (pd.DataFrame): Dataframe containing feature data
        
    ''' 
    tfidf, _ = create_tfidf_embeddings(collection, tfidf=tfidf, name='collection')
    create_tfidf_embeddings(queries, tfidf=tfidf, name='query')
    glove, _ = create_glove_embeddings(collection, glove=glove, name='collection')
    create_glove_embeddings(queries, glove=glove, name='query')
    bert, _ = create_bert_embeddings(collection, bert=bert, name='collection')
    create_bert_embeddings(queries, bert=bert, name='query')
    w2v, _ = create_w2v_embeddings(collection, w2v=w2v, name='collection')
    create_w2v_embeddings(queries, w2v=w2v, name='query')
    create_w2v_embeddings_tf_idf_weighted(collection, w2v=w2v, name = "collection")
    create_w2v_embeddings_tf_idf_weighted(queries, w2v=w2v, name = "query")
    features = create_w2v_feature(features, collection, queries)
    features = create_w2v_tfidf_feature(features, collection, queries)
    features = create_tfidf_feature(features, collection, queries)
    features = create_bert_feature(features, collection, queries)
    features = create_glove_feature(features, collection, queries)
    features = create_jaccard_feature(features, collection, queries)
    features = create_sentence_features(features, collection, queries)
    features = create_interpretation_features(features, collection, queries)
    features = create_BM2_feature(features, collection, queries)
    return create_POS_features(features, collection, queries)


def create_tfidf_embeddings(data: pd.DataFrame, tfidf=None, name: str = ''):
    ''' Creates tfidf embeddings

    Args:
        data (pd.DataFrame): Dataframe containing data to be embedded
        tfidf (str): Creates new object of class tfidf if None
        name (str): Adds string to name of the .pkl file created and stored of the data data frame

    Returns:
        tfidf (TFIDF object): Object of class TFIDF
        data (pd.DataFrame): Dataframe data with new column "preprocessed" appended 
        
    ''' 
    if tfidf is None:
        tfidf = TFIDF()
        tfidf.fit(
            data['preprocessed']
        )
    data['tfidf'] = tfidf.transform(
        data['preprocessed'],
        f"data/embeddings/tfidf_{name}_embeddings.pkl")

    return tfidf, data


def create_glove_embeddings(data: pd.DataFrame, glove=None, name: str = ''):
    ''' Creates glove embeddings

    Args:
        data (pd.DataFrame): Dataframe containing data to be embedded
        glove (str): Creates new object of class Glove if None
        name (str): Adds string to name of the .pkl file created and stored of the data Dataframe

    Returns:
        glove (Glove object): Object of class Glove
        data (pd.DataFrame): Dataframe data with new column "preprocessed" appended 
        
    ''' 
    if glove is None:
        glove = Glove()

    data['glove'] = glove.transform(
        data['preprocessed'],
        f"data/embeddings/glove_{name}_embeddings.pkl")

    return glove, data

def create_glove_embeddings_tf_idf_weighted(data: pd.DataFrame, glove=None, name: str = ''):
    ''' Creates tfidf weighted glove embeddings

    Args:
        data (pd.DataFrame): Dataframe containing data to be embedded
        glove (str): Creates new object of class Glove if None
        name (str): Adds string to name of the .pkl file created and stored of the data Dataframe

    Returns:
        glove (Glove object): Object of class Glove
        data (pd.DataFrame): Dataframe data with new column "glove_tfidf" appended 
        
    ''' 
    if glove is None:
        glove = Glove()

    data['glove_tfidf'] = glove.transform_tfidfweighted(
        data['preprocessed'], data['tfidf'],
        f"data/embeddings/glove_tf_idf_{name}_embeddings.pkl")

    return glove, data   


def create_bert_embeddings(data: pd.DataFrame, bert=None, name: str = ''):
    ''' Creates bert embeddings

    Args:
        data (pd.DataFrame): Dataframe containing data to be embedded
        bert (str): Creates new object of class Bert if None
        name (str): Adds string to name of the .pkl file created and stored of the data Dataframe

    Returns:
        bert (Bert object): Object of class Bert
        data (pd.DataFrame): Dataframe data with new column "preprocessed" appended 
        
    ''' 
    if bert is None:
        bert = Bert()

    column_name = ""
    if name == "collection":
        column_name = "Passage"
    if name == "query" or name == "query_test":
        column_name = "Query"

    data['bert'] = bert.transform(
        data[column_name],
        f"data/embeddings/bert_{name}_embeddings.pkl")

    return bert, data


def create_w2v_embeddings(data: pd.DataFrame, w2v=None, name: str = ''):
    ''' Creates word2vec embeddings

    Args:
        data (pd.DataFrame): Dataframe containing data to be embedded
        w2v (str): Creates new object of class word2vec if None
        name (str): Adds string to name of the .pkl file created and stored of the data Dataframe

    Returns:
        w2v (word2vec object): Object of class word2vec
        data (pd.DataFrame): Dataframe data with new column "preprocessed" appended 
        
    ''' 
    if w2v is None:
        w2v = word2vec()

    data['w2v'] = w2v.transform(data['preprocessed'],
                                f"data/embeddings/w2v_{name}_embeddings.pkl")

    return w2v, data

def create_w2v_embeddings_tf_idf_weighted(data: pd.DataFrame, w2v=None, name: str = ''):
    ''' Creates weighted tfidf word2vec embeddings

    Args:
        data (pd.DataFrame): Dataframe containing data to be embedded
        w2v (str): Creates new object of class word2vec if None
        name (str): Adds string to name of the .pkl file created and stored of the data Dataframe

    Returns:
        w2v (word2vec object): Object of class word2vec
        data (pd.DataFrame): Dataframe data with new column "w2v_tfidf" appended 
        
    ''' 
    if w2v is None:
        w2v = word2vec()

    data['w2v_tfidf'] = w2v.transform_tf_idf_weighted(data['preprocessed'], data['tfidf'], 
                                f"data/embeddings/w2v_tfidf_{name}_embeddings.pkl")

    return w2v, data

def create_w2v_feature(features: pd.DataFrame, collection: pd.DataFrame, queries: pd.DataFrame,
                       path_collection: str = 'data/embeddings/w2v_collection_embeddings.pkl',
                       path_query: str = 'data/embeddings/w2v_query_embeddings.pkl'):
    ''' Creates word2vec features (cosine, euclidean, manhattan)

    Args:
        features (pd.DataFrame): Dataframe containing feature data
        collection (pd.DataFrame): Dataframe containing collection data
        queries (pd.DataFrame): Dataframe containing queries data
        path_collection (str): Path to "w2v_collection_embeddings.pkl" file
        path_query (str): Path to "w2v_query_embeddings.pkl" file


    Returns:
        features (pd.DataFrame): Dataframe "features" with new columns "w2v_cosine", "w2v_euclidean", "w2v_manhattan" appended 
        
    ''' 
    embeddings = np.array(load(path_collection))
    embeddings_queries = np.array(load(path_query))

    features['w2v_cosine'] = features.progress_apply(lambda qrel:
                                                     cosine_similarity_score(embeddings_queries[
                                                                                 queries[
                                                                                     queries[
                                                                                         'qID'] == qrel.qID].index],
                                                                             embeddings[collection[
                                                                                 collection[
                                                                                     'pID'] == qrel.pID].index]),
                                                     axis=1)
    features['w2v_euclidean'] = features.progress_apply(lambda qrel:
                                                        euclidean_distance_score(embeddings_queries[
                                                                                     queries[
                                                                                         queries[
                                                                                             'qID'] == qrel.qID].index],
                                                                                 embeddings[
                                                                                     collection[
                                                                                         collection[
                                                                                             'pID'] == qrel.pID].index]),
                                                        axis=1)
    features['w2v_manhattan'] = features.progress_apply(lambda qrel:
                                                        manhattan_distance_score(embeddings_queries[
                                                                                     queries[
                                                                                         queries[
                                                                                             'qID'] == qrel.qID].index],
                                                                                 embeddings[
                                                                                     collection[
                                                                                         collection[
                                                                                             'pID'] == qrel.pID].index]),
                                                        axis=1)

    return features

def create_w2v_tfidf_feature(features: pd.DataFrame, collection: pd.DataFrame, queries: pd.DataFrame,
                       path_collection: str = 'data/embeddings/w2v_tfidf_collection_embeddings.pkl',
                       path_query: str = 'data/embeddings/w2v_tfidf_query_embeddings.pkl'):
    ''' Creates tfidf weighted word2vec features (cosine, euclidean, manhattan)

    Args:
        features (pd.DataFrame): Dataframe containing feature data
        collection (pd.DataFrame): Dataframe containing collection data
        queries (pd.DataFrame): Dataframe containing queries data
        path_collection (str): Path to "w2v_tfidf_collection_embeddings.pkl" file
        path_query (str): Path to "w2v_tfidf_query_embeddings.pkl" file


    Returns:
        features (pd.DataFrame): Dataframe "features" with new columns "w2v_tfidf_cosine", "w2v_tfidf_euclidean", "w2v_tfidf_manhattan" appended 
        
    ''' 
    embeddings = np.array(load(path_collection))
    embeddings_queries = np.array(load(path_query))

    features['w2v_tfidf_cosine'] = features.progress_apply(lambda qrel:
                                                     cosine_similarity_score(embeddings_queries[
                                                                                 queries[
                                                                                     queries[
                                                                                         'qID'] == qrel.qID].index],
                                                                             embeddings[collection[
                                                                                 collection[
                                                                                     'pID'] == qrel.pID].index]),
                                                     axis=1)
    features['w2v_tfidf_euclidean'] = features.progress_apply(lambda qrel:
                                                        euclidean_distance_score(embeddings_queries[
                                                                                     queries[
                                                                                         queries[
                                                                                             'qID'] == qrel.qID].index],
                                                                                 embeddings[
                                                                                     collection[
                                                                                         collection[
                                                                                             'pID'] == qrel.pID].index]),
                                                        axis=1)
    features['w2v_tfidf_manhattan'] = features.progress_apply(lambda qrel:
                                                        manhattan_distance_score(embeddings_queries[
                                                                                     queries[
                                                                                         queries[
                                                                                             'qID'] == qrel.qID].index],
                                                                                 embeddings[
                                                                                     collection[
                                                                                         collection[
                                                                                             'pID'] == qrel.pID].index]),
                                                        axis=1)

    return features

def create_tfidf_feature(features: pd.DataFrame, collection: pd.DataFrame, queries: pd.DataFrame,
                         path_collection: str = 'data/embeddings/tfidf_collection_embeddings.pkl',
                         path_query: str = 'data/embeddings/tfidf_query_embeddings.pkl'):
    ''' Creates tfidf features (cosine, euclidean, manhattan)

    Args:
        features (pd.DataFrame): Dataframe containing feature data
        collection (pd.DataFrame): Dataframe containing collection data
        queries (pd.DataFrame): Dataframe containing queries data
        path_collection (str): Path to "tfidf_collection_embeddings.pkl" file
        path_query (str): Path to "tfidf_query_embeddings.pkl" file


    Returns:
        features (pd.DataFrame): Dataframe "features" with new columns "tfidf_cosine", "tfidf_euclidean", "tfidf_manhattan" appended 
        
    ''' 
    embeddings = load(path_collection)
    embeddings_queries = load(path_query)

    features['tfidf_cosine'] = features.progress_apply(lambda qrel:
                                                       cosine_similarity_score(embeddings_queries[
                                                                                   queries[
                                                                                       queries[
                                                                                           'qID'] == qrel.qID].index],
                                                                               embeddings[collection[
                                                                                   collection[
                                                                                       'pID'] == qrel.pID].index]),
                                                       axis=1)
    features['tfidf_euclidean'] = features.progress_apply(lambda qrel:
                                                          euclidean_distance_score(embeddings_queries[
                                                                                       queries[
                                                                                           queries[
                                                                                               'qID'] == qrel.qID].index],
                                                                                   embeddings[
                                                                                       collection[
                                                                                           collection[
                                                                                               'pID'] == qrel.pID].index]),
                                                          axis=1)
    features['tfidf_manhattan'] = features.progress_apply(lambda qrel:
                                                          manhattan_distance_score(embeddings_queries[
                                                                                       queries[
                                                                                           queries[
                                                                                               'qID'] == qrel.qID].index],
                                                                                   embeddings[
                                                                                       collection[
                                                                                           collection[
                                                                                               'pID'] == qrel.pID].index]),
                                                          axis=1)

    return features


def create_glove_feature(features: pd.DataFrame, collection: pd.DataFrame, queries: pd.DataFrame,
                         path_collection: str = 'data/embeddings/glove_collection_embeddings.pkl',
                         path_query: str = 'data/embeddings/glove_query_embeddings.pkl'):
    ''' Creates glove features (cosine, euclidean, manhattan)

    Args:
        features (pd.DataFrame): Dataframe containing feature data
        collection (pd.DataFrame): Dataframe containing collection data
        queries (pd.DataFrame): Dataframe containing queries data
        path_collection (str): Path to "glove_collection_embeddings.pkl" file
        path_query (str): Path to "glove_query_embeddings.pkl" file


    Returns:
        features (pd.DataFrame): Dataframe "features" with new columns "glove_cosine", "glove_euclidean", "glove_manhattan" appended 
        
    ''' 
    embeddings = np.array(load(path_collection))
    embeddings_queries = np.array(load(path_query))

    features['glove_cosine'] = features.progress_apply(lambda qrel:
                                                       cosine_similarity_score(embeddings_queries[
                                                                                   queries[
                                                                                       queries[
                                                                                           'qID'] == qrel.qID].index],
                                                                               embeddings[
                                                                                   collection[
                                                                                       collection[
                                                                                           'pID'] == qrel.pID].index]),
                                                       axis=1)
    features['glove_euclidean'] = features.progress_apply(lambda qrel:
                                                          euclidean_distance_score(embeddings_queries[
                                                                                       queries[
                                                                                           queries[
                                                                                               'qID'] == qrel.qID].index],
                                                                                   embeddings[
                                                                                       collection[
                                                                                           collection[
                                                                                               'pID'] == qrel.pID].index]),
                                                          axis=1)
    features['glove_manhattan'] = features.progress_apply(lambda qrel:
                                                          manhattan_distance_score(embeddings_queries[
                                                                                       queries[
                                                                                           queries[
                                                                                               'qID'] == qrel.qID].index],
                                                                                   embeddings[
                                                                                       collection[
                                                                                           collection[
                                                                                               'pID'] == qrel.pID].index]),
                                                          axis=1)

    return features


def create_bert_feature(features: pd.DataFrame, collection: pd.DataFrame, queries: pd.DataFrame,
                        path_collection: str = 'data/embeddings/bert_collection_embeddings.pkl',
                        path_query: str = 'data/embeddings/bert_query_embeddings.pkl'):
    ''' Creates bert features (cosine, euclidean, manhattan)

    Args:
        features (pd.DataFrame): Dataframe containing feature data
        collection (pd.DataFrame): Dataframe containing collection data
        queries (pd.DataFrame): Dataframe containing queries data
        path_collection (str): Path to "bert_collection_embeddings.pkl" file
        path_query (str): Path to "bert_query_embeddings.pkl" file


    Returns:
        features (pd.DataFrame): Dataframe "features" with new columns "bert_cosine", "bert_euclidean", "bert_manhattan" appended 
        
    ''' 
    embeddings = np.array(load(path_collection))

    embeddings_queries = np.array(load(path_query))

    features['bert_cosine'] = features.progress_apply(lambda qrel:
                                                      cosine_similarity_score(embeddings_queries[
                                                                                  queries[
                                                                                      queries[
                                                                                          'qID'] == qrel.qID].index],
                                                                              embeddings[
                                                                                  collection[
                                                                                      collection[
                                                                                          'pID'] == qrel.pID].index]),
                                                      axis=1)
    features['bert_euclidean'] = features.progress_apply(lambda qrel:
                                                         euclidean_distance_score(embeddings_queries[
                                                                                      queries[
                                                                                          queries[
                                                                                              'qID'] == qrel.qID].index],
                                                                                  embeddings[
                                                                                      collection[
                                                                                          collection[
                                                                                              'pID'] == qrel.pID].index]),
                                                         axis=1)
    features['bert_manhattan'] = features.progress_apply(lambda qrel:
                                                         manhattan_distance_score(embeddings_queries[
                                                                                      queries[
                                                                                          queries[
                                                                                              'qID'] == qrel.qID].index],
                                                                                  embeddings[
                                                                                      collection[
                                                                                          collection[
                                                                                              'pID'] == qrel.pID].index]),
                                                         axis=1)

    return features


def create_jaccard_feature(features: pd.DataFrame, collection: pd.DataFrame, queries: pd.DataFrame):
    ''' Creates jaccard features for query-collection combinations

    Args:
        features (pd.DataFrame): Dataframe containing feature data
        collection (pd.DataFrame): Dataframe containing collection data
        queries (pd.DataFrame): Dataframe containing queries data

    Returns:
        features (pd.DataFrame): Dataframe "features" with new column "jaccard" appended 
        
    ''' 
    features['jaccard'] = features.progress_apply(
        lambda qrel: jaccard(collection[collection['pID'] == qrel['pID']]['preprocessed'].iloc[0],
                             queries[queries['qID'] == qrel['qID']]['preprocessed'].iloc[0]),
        axis=1)

    return features


def create_sentence_features(features: pd.DataFrame, collection: pd.DataFrame, queries: pd.DataFrame):
    ''' Creates sentence features for query-collection combinations (words_difference, words_rel_difference, char_difference, char_rel_difference)

    Args:
        features (pd.DataFrame): Dataframe containing feature data
        collection (pd.DataFrame): Dataframe containing collection data
        queries (pd.DataFrame): Dataframe containing queries data

    Returns:
        features (pd.DataFrame): Dataframe "features" with new columns "words_doc", "words_query", "words_difference", "words_rel_difference"
        "char_doc", "char_query", "char_difference", "char_rel_difference" appended 
        
    ''' 
    features['words_doc'] = features.progress_apply(
        lambda qrel: words(collection[collection['pID'] == qrel['pID']]['Passage'].iloc[0]),
        axis=1)
    features['words_query'] = features.progress_apply(
        lambda qrel: words(queries[queries['qID'] == qrel['qID']]['Query'].iloc[0]),
        axis=1)
    features['words_difference'] = features.progress_apply(
        lambda qrel: difference(qrel['words_doc'], qrel['words_query']),
        axis=1)
    features['words_rel_difference'] = features.progress_apply(
        lambda qrel: relative_difference(qrel['words_doc'], qrel['words_query']),
        axis=1)

    features['char_doc'] = features.progress_apply(
        lambda qrel: characters(collection[collection['pID'] == qrel['pID']]['Passage'].iloc[0]),
        axis=1)
    features['char_query'] = features.progress_apply(
        lambda qrel: characters(queries[queries['qID'] == qrel['qID']]['Query'].iloc[0]),
        axis=1)
    features['char_difference'] = features.progress_apply(
        lambda qrel: difference(qrel['char_doc'], qrel['char_query']),
        axis=1)
    features['char_rel_difference'] = features.progress_apply(
        lambda qrel: relative_difference(qrel['char_doc'], qrel['char_query']),
        axis=1)

    return features


def create_interpretation_features(features: pd.DataFrame, collection: pd.DataFrame, queries: pd.DataFrame):
    ''' Creates interpretation features for query and collection data (subjectivity, polarity)

    Args:
        features (pd.DataFrame): Dataframe containing feature data
        collection (pd.DataFrame): Dataframe containing collection data
        queries (pd.DataFrame): Dataframe containing queries data

    Returns:
        features (pd.DataFrame): Dataframe "features" with new columns "subjectivity_doc", "polarity_doc", "subjectivity_query", "polarity_query" appended 
        
    ''' 
    features['subjectivity_doc'] = features.progress_apply(
        lambda qrel: subjectivity(collection[collection['pID'] == qrel['pID']]['Passage'].iloc[0]),
        axis=1)
    features['polarity_doc'] = features.progress_apply(
        lambda qrel: polarisation(collection[collection['pID'] == qrel['pID']]['Passage'].iloc[0]),
        axis=1)

    features['subjectivity_query'] = features.progress_apply(
        lambda qrel: subjectivity(queries[queries['qID'] == qrel['qID']]['Query'].iloc[0]),
        axis=1)
    features['polarity_query'] = features.progress_apply(
        lambda qrel: polarisation(queries[queries['qID'] == qrel['qID']]['Query'].iloc[0]),
        axis=1)

    return features


def create_POS_features(features: pd.DataFrame, collection: pd.DataFrame, queries: pd.DataFrame):
    ''' Creates Part of Speech features for query and collection data (nouns, adjectives, verbs)

    Args:
        features (pd.DataFrame): Dataframe containing feature data
        collection (pd.DataFrame): Dataframe containing collection data
        queries (pd.DataFrame): Dataframe containing queries data

    Returns:
        features (pd.DataFrame): Dataframe "features" with new columns "doc_nouns", "doc_adjectives", "doc_verbs", "query_nouns",
        "query_adjectives", "query_verbs" appended 
        
    ''' 
    pos = features.progress_apply(
        lambda qrel: POS(collection[collection['pID'] == qrel['pID']]['Passage'].iloc[0]),
        axis=1)
    features['doc_nouns'] = [tag[0] for tag in pos]
    features['doc_adjectives'] = [tag[1] for tag in pos]
    features['doc_verbs'] = [tag[2] for tag in pos]

    pos = features.progress_apply(
        lambda qrel: POS(queries[queries['qID'] == qrel['qID']]['Query'].iloc[0]),
        axis=1)
    features['query_nouns'] = [tag[0] for tag in pos]
    features['query_adjectives'] = [tag[1] for tag in pos]
    features['query_verbs'] = [tag[2] for tag in pos]

    return features


def create_BM2_feature(features: pd.DataFrame, collection: pd.DataFrame, queries: pd.DataFrame):
    ''' Creates BM25 features for query-collection combinations

    Args:
        features (pd.DataFrame): Dataframe containing feature data
        collection (pd.DataFrame): Dataframe containing collection data
        queries (pd.DataFrame): Dataframe containing queries data

    Returns:
        features (pd.DataFrame): Dataframe "features" with new column "bm25" appended 
        
    ''' 
    bm25 = BM25().fit(collection['preprocessed'])
    features['bm25'] = features.progress_apply(
        lambda qrel: bm25.predict_proba(queries[queries['qID'] == qrel['qID']]['preprocessed'].iloc[0],
                                        collection[collection['pID'] == qrel['pID']]['preprocessed'].iloc[0]),
        axis=1)

    return features
