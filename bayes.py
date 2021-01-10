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
        # 学習された単語の集合
        self.vocabularies = set()
        self.word_count = {}

        self.category_count = {}
        self.category_words = {}

    def fit(self, category: str, texts: list):
        if category in self.category_count:
            raise Exception("%s is already registered" % category)

        self.category_count[category] = len(texts)

        concatenated_text = " ".join(texts)
        words = self.to_words(concatenated_text)
        counts = self.count_words(words)

        # 最低でもn回出たら学習する
        min_word_count = 1
        if min_word_count >= 2:
            counts = {k: v for k, v in counts.items() if v >= min_word_count}
            words = [w for w in words if w in set(counts.keys())]

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
        text = self._normalize_document(text)
        words = self._separate_text(text)
        words = self._remove_dc_words(words)
        return words

    def _normalize_document(self, text: str) -> str:
        t = unicodedata.normalize("NFKC", text)
        t = t.lower()
        t = re.sub(r'\s+', r' ', t)
        return t

    def _remove_dc_words(self, words: list) -> list:
        discard_words = r"(\s+)"  # |[,.()\[\]（）「」'\"、。]
        words = [w for w in words if not re.match(discard_words, w)]
        return words

    def _separate_text(self, text: str) -> list:
        return wakati.parse(text).split()

    def classify(self, text: str) -> dict:
        words = self.to_words(text)
        counts = self.count_words(words)

        category_scores = {}
        for category in self.category_count.keys():
            score = np.log(self._category_p(category))
            score += self._calc_text_score(category, counts)
            category_scores[category] = score

        base = max(category_scores.values())
        p = {k: 10 ** (v - base)
             for k, v in category_scores.items()}

        n = sum(p.values())
        p = {k: v / n for k, v in p.items()}
        return p

    def _category_p(self, category: str) -> float:
        return self.category_count[category] / sum(self.category_count.values())

    def _calc_text_score(self, category: str, word_counts: dict) -> int:
        evidence = float(
            sum(self.category_words[category].values())
            + len(self.vocabularies)
        )

        score = 0
        for word in word_counts.keys():
            p = (self._get_word_count(category, word) + 1.0) / evidence
            score += np.log(p) * word_counts[word]

        return score

    def _get_word_count(self, category: str, word: str) -> float:
        if word in self.category_words[category]:
            return float(self.category_words[category][word])
        return 0.0


def main():
    test_count = 1570

    df = pd.read_table("./SMSSpamCollection.csv", sep='\t',
                       header=None, names=['class', 'text'])
    df = df.sample(frac=1).reset_index(drop=True)

    test_df = df[0:test_count]
    df = df[test_count:].reset_index(drop=True)

    obj = NaiveBayes()
    obj.fit("spam", df["text"][df["class"] == "spam"])
    obj.fit("ham", df["text"][df["class"] == "ham"])

    missed_spam = 0
    captured_spam = 0

    passed_ham = 0
    blocked_ham = 0

    st = time.time()

    N = 0
    for doc in test_df.iloc:
        N += 1

        p = obj.classify(doc["text"])
        spam_rate = p["spam"]

        is_spam = spam_rate > 0.5

        if doc["class"] == "spam":
            if is_spam:
                captured_spam += 1
            else:
                missed_spam += 1
        else:
            if is_spam:
                blocked_ham += 1
            else:
                passed_ham += 1

    print("t=", time.time() - st)

    print("Train: ", len(df))
    print("SPAM count", len(test_df["text"][test_df["class"] == "spam"]))
    print("HAM count", len(test_df["text"][test_df["class"] == "ham"]))
    print()

    print("N:", N)
    print("正解率", round((passed_ham + captured_spam) / N, 3))
    print("スパム正解率", "{}/{}".format(captured_spam, (captured_spam + missed_spam)),
          round(captured_spam / (captured_spam + missed_spam), 3))
    print("ハムの正解率", "{}/{}".format(passed_ham, (passed_ham + blocked_ham)),
          round(passed_ham / (passed_ham + blocked_ham), 3))


if __name__ == "__main__":
    for _ in range(10):
        main()
        print()
        break
