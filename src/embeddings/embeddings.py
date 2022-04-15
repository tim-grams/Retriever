from gensim.models import Word2Vec

# Create word2vec embedding with preprocessed index terms occuring at least min_count times
def create_word2vec(text, min_count):
	return Word2Vec(text, min_count=min_count)
	
# Create dict with unique words that exists at least min_count times	
def word2vec_get_unique(w2v_object):
	return w2v_object.wv.vocab

# Returns vector representation of specific word.	
def word2vec_vec_repr(w2v_object, word):
	return w2v_object.wv[word]

# Returns similarity of terms in word2vec model to specified word	
def word2vec_get_similar(w2v_object, word):
	return w2v_object.wv.most_similar(word)