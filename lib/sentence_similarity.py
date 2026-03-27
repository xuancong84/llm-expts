#!/usr/bin/env python

# pip install sentence-transformers

from sentence_transformers import SentenceTransformer, util, CrossEncoder

M_uncased = None
def sentence_similarity_uncased(s1, s2):
	"""
		Case-insensitive (better for one sentence vs many)
		It computes the cosine similarity between everyone in s1 and everyone in s2.
	"""
	global M_uncased
	if M_uncased is None:
		M_uncased = SentenceTransformer("all-MiniLM-L6-v2")
	e1 = M_uncased.encode(s1, convert_to_tensor=True, show_progress_bar=False)
	e2 = M_uncased.encode(s2, convert_to_tensor=True, show_progress_bar=False)
	return util.cos_sim(e1, e2).tolist()


M_cased = None
def sentence_similarity_cased(s1, s2):
	"""
		Case-sensitive (better for one sentence vs many)
		It computes the cosine similarity between everyone in s1 and everyone in s2.
	"""
	global M_cased
	if M_cased is None:
		m = 'sentence-transformers/stsb-bert-base' or "sentence-transformers/distiluse-base-multilingual-cased"
		M_cased = SentenceTransformer(m)
	e1 = M_cased.encode(s1, convert_to_tensor=True, show_progress_bar=False)
	e2 = M_cased.encode(s2, convert_to_tensor=True, show_progress_bar=False)
	ret = util.cos_sim(e1, e2).tolist()
	return ret[0][0] if type(s1)==str==type(s2) else ret



M_cross = None
def sentence_similarity_crossEncoder(s1, s2):
	"""
		Case-sensitive (better for sentence pairs)
		It computes the cross-encodersimilarity between all pairs in s1 and s2.
	"""
	global M_cross
	if M_cross is None:
		M_cross = CrossEncoder("cross-encoder/stsb-roberta-base")
	pairs = [s1, s2] if isinstance(s1, str) else [[s1_i, s2_i] for s1_i, s2_i in zip(s1, s2)]
	scores = M_cross.predict(pairs, show_progress_bar=False)
	return scores.tolist()


if __name__ == "__main__":
	a = sentence_similarity_uncased("Hello", "hello")
	b = sentence_similarity_cased("Hello", "hello")
	c = sentence_similarity_crossEncoder("This is an elephant.", "Here got one elephant.")
	aa=5
	