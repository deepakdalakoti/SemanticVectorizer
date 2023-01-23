"""
    A simple wrapper around CountVectorizer which counts semantically similar tokens
    SenetenceTransformer embeddings are used to identify semantically similar tokens
    Agglomerative clustering is used to group similar terms

"""


import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import CountVectorizer


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
        self.cluster_model = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="complete",
            distance_threshold=1 - self.similarity_threshold,
        )

    def fit(self, raw_documents, y=None):
        # as per sklearn, fit calls fit_transform
        self.fit_transform(raw_documents)
        return self

    def transform(self, raw_documents):
        # Transform a given list of documents using vocabulary learned during fit
        X = super().transform(raw_documents)
        # We only care about aggregate values and not per document values
        counts = X.sum(axis=0)

        terms = np.array(list(self.vocabulary_.keys()))
        indices = np.array(list(self.vocabulary_.values()))
        inverse_vocab = terms[np.argsort(indices)]
        # Tokens and counts
        tokens = inverse_vocab[counts.nonzero()[1].ravel()]
        counts = counts[0, counts.nonzero()[1].ravel()].tolist()[0]

        # Compute embeddings, perhaps cache?
        embeddings = self.embedding_model.encode(tokens)
        labels = self.cluster_model.fit_predict(embeddings)
        vals = {"term": tokens, "labels": labels, "counts": counts}
        df_out = pd.DataFrame(vals)
        df_out = (
            df_out.groupby(["labels"])
            .agg({"term": ",".join, "counts": sum})
            .reset_index()
        )
        return df_out

    def fit_transform(self, raw_documents, y=None):
        # Fit CountVectorizer and get counts of vocabulary
        X = super().fit_transform(raw_documents)
        #Sort by vocabulary index, because counts are given in this order
        terms = [k for k, _ in sorted(self.vocabulary_.items(), key=lambda item: item[1])]
        # Compute embeddings, potentially could add dimensionally reduction like BerTopic
        embeddings = self.embedding_model.encode(terms)
        # Fit clustering model to identify similar tokens
        self.cluster_model.fit(embeddings)
        # Count of tokens in all documents
        counts = X.sum(axis=0).tolist()
        # make a df_out
        vals = {
            "term": terms,
            "labels": self.cluster_model.labels_,
            "counts": counts[0],
        }
        df_out = pd.DataFrame(vals)
        # group counts and terms by cluster labels
        df_out = (
            df_out.groupby(["labels"])
            .agg({"term": ",".join, "counts": sum})
            .reset_index()
        )
        return df_out
