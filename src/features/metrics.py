from sklearn.metrics.pairwise import cosine_similarity


def cosine_similarity_score(embedding_1, embedding_2):
    return cosine_similarity(embedding_1, embedding_2)