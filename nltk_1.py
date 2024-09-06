# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn.feature_extraction.text import CountVectorizer

doc_text=["teachers teach many students","Many engineers work for society","teachers and students build society"]

vectorizer=CountVectorizer()
vectorizer.fit(doc_text)

vector=vectorizer.transform(doc_text)

print("vocabulary :",vectorizer.vocabulary_)

print("Encoded doc_text :")
print(vector.toarray())