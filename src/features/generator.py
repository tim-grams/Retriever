import pandas as pd
from tqdm import tqdm
import logging
import numpy as np
from src.embeddings.tfidf import TFIDF
from src.embeddings.glove import Glove
from src.features.features import cosine_similarity_score, euclidean_distance_score, manhattan_distance_score, jaccard, \
    words, relative_difference, characters, difference, subjectivity, polarisation, POS
from src.utils.utils import load
from src.features.bm25 import BM25

tqdm.pandas()
LOGGER = logging.getLogger('generator')


def create_all(features: pd.DataFrame, collection: pd.DataFrame, queries: pd.DataFrame, tfidf=None):
    tfidf, _ = create_tfidf_embeddings(collection, tfidf=tfidf, name='collection')
    create_tfidf_embeddings(queries, tfidf=tfidf, name='query')
    glove, _ = create_glove_embeddings(collection, glove=glove, name='collection')
    create_glove_embeddings(queries, glove=glove, name='query')
    features = create_tfidf_feature(features, collection, queries)
    features = create_glove_feature(features, collection, queries)
    features = create_jaccard_feature(features, collection, queries)
    features = create_sentence_features(features, collection, queries)
    features = create_interpretation_features(features, collection, queries)
    features = create_BM2_feature(features, collection, queries)
    return create_POS_features(features, collection, queries)


def create_tfidf_embeddings(data: pd.DataFrame, tfidf=None, name: str = ''):
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
        if glove is None:
            glove = Glove()

        data['glove'] = glove.transform(
            data['preprocessed'],
            f"data/embeddings/glove_{name}_embeddings.pkl")

        return glove, data


def create_tfidf_feature(features: pd.DataFrame, collection: pd.DataFrame, queries: pd.DataFrame,
                         path_collection: str = 'data/embeddings/tfidf_collection_embeddings.pkl',
                         path_query: str = 'data/embeddings/tfidf_query_embeddings.pkl'):
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

def create_jaccard_feature(features: pd.DataFrame, collection: pd.DataFrame, queries: pd.DataFrame):
    features['jaccard'] = features.progress_apply(
        lambda qrel: jaccard(collection[collection['pID'] == qrel['pID']]['preprocessed'].iloc[0],
                             queries[queries['qID'] == qrel['qID']]['preprocessed'].iloc[0]),
        axis=1)

    return features


def create_sentence_features(features: pd.DataFrame, collection: pd.DataFrame, queries: pd.DataFrame):
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
    bm25 = BM25().fit(collection['preprocessed'])
    features['bm25'] = features.progress_apply(
        lambda qrel: bm25.predict_proba(queries[queries['qID'] == qrel['qID']]['preprocessed'].iloc[0],
                                        collection[collection['pID'] == qrel['pID']]['preprocessed'].iloc[0]),
        axis=1)

    return features
