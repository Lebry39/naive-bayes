import pandas as pd
import time
import numpy as np
import math
import re
import unicodedata
import MeCab  # https://pypi.org/project/mecab-python3/

wakati = MeCab.Tagger("-Owakati")

class NaiveBayes:
    def __init__(self):
        self.vocabularies = set()
        self.category_count = {}
        self.category_words = {}
        self.word_count = {}
            
    def fit(self, category, texts: list):
        if category in self.category_count:
            raise Exception("%s is already registered" % category)
        
        self.category_count[category] = len(texts)
        
        concatenated_text = " ".join(texts)
        words = self.to_words(concatenated_text)
        counts = self.count_words(words)
        
        vocabulary = set(words)
        self.vocabularies |= vocabulary
        self.category_words[category] = counts
        
        for w in vocabulary:
            if w not in self.word_count:
                self.word_count[w] = 0
            self.word_count[w] += self.category_words[category][w]
    
    def count_words(self, words: list) -> dict:
        counts = {k: 0 for k in set(words)}
        for w in words:
            counts[w] += 1
        
        return counts
    
    def to_words(self, text: str) -> list:
        words = self._normalize_document(text)
        words = self._separate_text(words)
        return words
            
    def _normalize_document(self, text: str) -> str:
        t = unicodedata.normalize("NFKC", text)
        t = t.lower()
        t = re.sub('\s+', ' ', t)
        return t
    
    def _separate_text(self, text: str) -> list:
        return wakati.parse(text).split()
    
    
    def classify(self, text: str):
        words = self.to_words(text)
        counts = self.count_words(words)

        category_scores = {}
        for category in self.category_count.keys():
            score = np.log(self._category_p(category))
            score += self._document_score(category, counts)
            category_scores[category] = score

        n = sum(category_scores.values())
        return {k: 1 - (v / n) for k, v in category_scores.items()}
    
    def _document_score(self, category, word_counts):
        score = 0
        for word in word_counts.keys():
            p = self._word_p(category, word)
            score += np.log(p) * word_counts[word]
        return score
            
    def _word_p(self, category, word):
        return float(
            (self._get_word_count(category, word) + 1.0) 
            / (sum(self.category_words[category].values()) + float(len(self.vocabularies)))
        )            
            
    def _get_word_count(self, category, word):
        if word in self.category_words[category]:
            return float(self.category_words[category][word])
        return 0.0
    
    
    def _category_p(self, category):
        return self.category_count[category] / sum(self.category_count.values())

test_count = 1000

df = pd.read_table("./SMSSpamCollection.csv", sep='\t', header=None, names=['class', 'text'])
df = df.sample(frac=1).reset_index(drop=True)
# display(df)

test_df = df[0:test_count]
df = df[test_count:].reset_index(drop=True)

obj = NaiveBayes()
obj.fit("spam", df["text"][df["class"] == "spam"])
obj.fit("ham", df["text"][df["class"] == "ham"])



ok_count = 0
ng_count = 0

st = time.time()

N = 0
for doc in test_df.iloc:
    try:
        p = obj.classify(doc["text"])
        spam_rate = p["spam"]
    except:
        continue
        
    N += 1
#     print(spam_rate)
    if spam_rate < 0.5:
        continue
        
    if doc["class"] == "spam":
        ok_count += 1
    else:
        ng_count += 1
        

print("t=", time.time() - st)
print()

print("Train: ", len(df))
print("SPAM count",len(test_df["text"][test_df["class"] == "spam"]))
print("HAM count",len(test_df["text"][test_df["class"] == "ham"]))
print()

print("N:", N)
print("OK:", ok_count)
print("NG:", ng_count)
print(ok_count / (ok_count + ng_count + 0.00001))
print(ok_count / N)