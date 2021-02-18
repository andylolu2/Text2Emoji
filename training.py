from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('paraphrase-distilroberta-base-v1')

#Our sentences we like to encode
sentences = [
    'I go to school by bus',
    'I went to school with bus', 
    'Chocolate cakes tastes amazing'
    ]

#Sentences are encoded by calling model.encode()
sentence_embeddings = model.encode(sentences)

dist1 = np.linalg.norm(sentence_embeddings[0]-sentence_embeddings[1])
dist = np.linalg.norm(sentence_embeddings[0]-sentence_embeddings[2])

print(dist1, dist)