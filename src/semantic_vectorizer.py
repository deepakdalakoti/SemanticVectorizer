from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

class SemanticCountVectorizer(CountVectorizer):
    """
        A simple wrapper around CountVectorizer which counts semantically similar tokens
        SenetenceTransformer embeddings are used to identify semantically similar tokens
        Agglomerative clustering is used to group similar terms

    """
    def __init__(self, embedding_model_name, similarity_threshold=0.8, **kwargs):

        super().__init__(**kwargs)
        self.embedding_model_name = embedding_model_name
        self.similarity_threshold = similarity_threshold
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.cluster_model = AgglomerativeClustering(n_clusters=None,metric='cosine',linkage='complete',distance_threshold=1-self.similarity_threshold)

    
    def fit(self, data):
        #as per sklearn, fit calls fit_transform
        self.fit_transform(data)
        return self

    def transform(self, data):
        #Transform a given list of documents using vocabulary learned during fit
        X = super().transform(data)
        #We only care about aggregate values and not per document values
        counts = X.sum(axis=0)

        terms = np.array(list(self.vocabulary_.keys()))
        indices = np.array(list(self.vocabulary_.values()))
        inverse_vocab = terms[np.argsort(indices)]
        #Tokens and counts        
        tokens = inverse_vocab[counts.nonzero()[1].ravel()]
        counts = counts[0,counts.nonzero()[1].ravel()].tolist()[0]

        #Compute embeddings, perhaps cache?
        embeddings = self.embedding_model.encode(tokens)
        labels = self.cluster_model.fit_predict(embeddings)
        vals = {'term':tokens, 'labels':self.cluster_model.labels_, 'counts':counts}
        df = pd.DataFrame(vals)
        df = df.groupby(['labels']).agg({'term': lambda x: ",".join(x),'counts': sum}).reset_index()
        return df

        
    def fit_transform(self, data):
        #Fit CountVectorizer and get counts of vocabulary
        X = super().fit_transform(data)
        #Compute embeddings, potentially could add dimensionally reduction like BerTopic
        embeddings = self.embedding_model.encode(list(self.vocabulary_.keys()))
        #Fit clustering model to identify similar tokens
        self.cluster_model.fit(embeddings)
        #Count of tokens in all documents
        counts = X.sum(axis=0).tolist()
        #make a df
        vals = {'term':list(self.vocabulary_.keys()), 'labels':self.cluster_model.labels_, 'counts':counts[0]}
        df = pd.DataFrame(vals)
        #group counts and terms by cluster labels
        df = df.groupby(['labels']).agg({'term': lambda x: ",".join(x),'counts': sum}).reset_index()
        return df