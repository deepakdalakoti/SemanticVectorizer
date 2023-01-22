### Wrapper around CountVectorizer to count phrases based on semantic similarity

###Usage

`
from semantic_vectorizer import SemanticCountVectorizer 
sentences = ['this is a test sentences','this is another test sentence']
svect = SemanticVectorizer(embedding_model_name='all-MiniLM-L6-v2', similarity_threshold=0.7, ngram_range=(3,3))
counts=svect.fit_transform(sentences)

`
