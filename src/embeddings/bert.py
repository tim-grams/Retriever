from sentence_transformers import SentenceTransformer, util
import nltk
import pandas as pd
from tqdm import tqdm
from src.utils.utils import check_path_exists, save, load
import os



class Bert(object):

    def __init__(self):
        self.model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1") #Not the best model in terms of similarity scores but only a bit worse then the best one but only 20% the Size
        #nltk.download("punkt")

    #Dont Preprocess Texts beforehand! 
    def transform(self, raw_texts: pd.Series, store: str = None):
        bert_vec = []

        for text in tqdm(raw_texts):

            #In Testing the Similairity Scores of taking the whole passage as a Sentence or cutting the passage up were negligeble. So for now i take the Passage as 1 Sentence. So its faster
        
            #Splits text into sentences and cuts off after the second sentence (Bert can only handle 1 or 2 sentences)
            #sentences = nltk.tokenize.sent_tokenize(text) 
            #if len(sentences) > 2:
            #    sentences = sentences[0:2]
            #    embedding = self.model.encode(sentences)
            #else:
            embedding = self.model.encode(text)    
            bert_vec.append(embedding)
        
        if store is not None:
            check_path_exists(os.path.dirname(store))
            save(bert_vec, store)
            
        return bert_vec


        

            
            

            
"""


bert = Bert()

query_embedding = bert.model.encode('How big is London')
passage_embedding = bert.model.encode('London has 9,787,426 inhabitants at the 2011 census. London is known for its finacial district. London is a nice City')

print(passage_embedding)
print(type(passage_embedding))

print("Similarity:", util.dot_score(query_embedding, passage_embedding)[0][0].numpy())
"""


"""passage = ["London has 9,787,426 inhabitants at the 2011 census. London is known for its finacial district", "The President of the United States is Joe Biden" ,"Ketchup consists of mostly Tomato and Sugar"]
query = ["How big is London. What is it", "Who is the President of the United States", ]
seriespassage = pd.Series(passage)
seriesquery = pd.Series(query)
passageembedding = bert.transform(passage)
queryembedding = bert.transform(query)

print(util.normalize_embeddings(passageembedding[0]))
print(util.normalize_embeddings(queryembedding[0]))

#print("Similarity 1 :", util.dot_score(passageembedding[0], queryembedding[0]))
#print("Similarity 2 :", util.dot_score(passageembedding[1], queryembedding[1]))
#print("Similarity 3 :", util.dot_score(passageembedding[2], queryembedding[1]))
"""