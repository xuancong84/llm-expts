#!/usr/bin/env python

# pip install sentence-transformers

### 1a. Case-insensitive (better for one sentence vs many)
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

e1 = model.encode("Apple", convert_to_tensor=True)
e2 = model.encode("apple", convert_to_tensor=True)

print(util.cos_sim(e1, e2).item())




### 1b. Case-sensitive (better for one sentence vs many)
from sentence_transformers import SentenceTransformer, util

# Use a cased model path, not an uncased one
model = SentenceTransformer("google-bert/bert-base-cased")

s1 = "Apple released new devices."
s2 = "apple released new devices."

emb1 = model.encode(s1, convert_to_tensor=True)
emb2 = model.encode(s2, convert_to_tensor=True)

score = util.cos_sim(emb1, emb2).item()
print(score)



### 2. Cross-Encoder (Case-sensitive) (better for sentence pairs)
from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/stsb-roberta-base")

pairs = [["Apple released new devices.","apple released new devices."]]

scores = model.predict(pairs)
print(scores[0])
