### Wrapper around CountVectorizer to count phrases based on semantic similarity
---
SemanticCountVectorizer extends CountVectorizer by counting the frequency of semantically similar tokens. This is useful when searching for frequency of similar but not exact terms.

SemanticCountVectorizer uses sentence_transformers to compute embedding of n-grams and then clusters similar terms using Agglomerative Clustering using cosine similarity as the similarity metric

---
### Usage

```
from semantic_vectorizer import SemanticCountVectorizer 
sentences = ['this is a test sentences','this is another test sentence']
svect = SemanticVectorizer(embedding_model_name='all-MiniLM-L6-v2', similarity_threshold=0.7, ngram_range=(3,3))
counts=svect.fit_transform(sentences)
```

### Arguments
- embedding_model_name: Name of the sentence_transformer pretrained model to use. Refer to [package website](https://www.sbert.net/docs/pretrained_models.html) for a list of available models
- similarity_threshold: value between 0-1 which indicates minimum cosine similarity needed to group tokens.
- Arguments of sklearn's CountVectorizer as listed [here](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)