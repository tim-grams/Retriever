from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances


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
