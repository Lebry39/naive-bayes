import pandas as pd
import time
import numpy as np
import math
import re
import unicodedata
import MeCab  # https://pypi.org/project/mecab-python3/

# sudo apt-get install mecab mecab-ipadic-utf8
# -d /var/lib/mecab/dic/ipadic-utf8
wakati = MeCab.Tagger("-Owakati -d /var/lib/mecab/dic/ipadic-utf8")


class NaiveBayes:
    def __init__(self):
        # ラプラススムージングのパラメーター
        self.smoothing_a = 0.01
        self.smoothing_b = 0.02

        # 登録された単語の集合
        self._vocabularies = set()

        # ワードの出現数
        # {"wordA": 15, "wordB": 2, ...}
        self._word_count = {}

        # カテゴリごとのドキュメントの登録数
        # {"CategoryA": 16, "CategoryB": 9}
        self._category_count = {}

        # カテゴリごとのワードの出現数
        # {"CategoryA": {"wordA": 12, "wordB": 1}, ...}
        self._category_words = {}

        # 事前に計算されたワードスコア
        # {"CategoryA": {"wordA": -902, "wordB": -122}, ...}
        self._calculated_word_scores = {}

    def fit(self, category: str, texts: list):
        if category in self._category_count:
            raise Exception("%s is already registered" % category)

        # キャッシュを初期化
        self._calculated_word_scores = None
        self._category_count[category] = len(texts)

        words = []
        for text in texts:
            words += self.to_words(text)
        counts = self.count_words(words)

        vocabulary = set(words)
        self._vocabularies |= vocabulary
        self._category_words[category] = counts

        for w in vocabulary:
            if w not in self._word_count:
                self._word_count[w] = 0
            self._word_count[w] += self._category_words[category][w]

    def pre_calculate_word_score(self):
        self._calculated_word_scores = {}

        for category in self._category_count:
            evidence = float(
                sum(self._category_words[category].values())
                + len(self._vocabularies) * self.smoothing_b
            )
            self._calculated_word_scores[category] = {}

            for word in self._category_words[category]:
                word_score = np.log10((
                    self._get_word_count(category, word) +
                    self.smoothing_a
                ) / evidence)

                self._calculated_word_scores[category][word] = word_score

            nohit_score = np.log10(self.smoothing_a / evidence)
            self._calculated_word_scores[category][None] = nohit_score

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
        t = re.sub(r'\s+', r' ', text)
        t = unicodedata.normalize("NFKC", t)
        t = t.lower()
        return t

    def _remove_dc_words(self, words: list) -> list:
        discard_words = r"\s+"
        words = [w for w in words if not re.match(discard_words, w)]
        return words

    def _separate_text(self, text: str) -> list:
        return wakati.parse(text).split()

    def classify(self, text: str) -> dict:
        if self._calculated_word_scores is None:
            self.pre_calculate_word_score()

        words = self.to_words(text)
        counts = self.count_words(words)

        category_scores = {}
        for category in self._category_count:
            score = np.log10(self._category_p(category))
            score += self._calc_text_score(category, counts)
            category_scores[category] = score

        base = max(category_scores.values())
        p = {k: 10 ** (v - base)
             for k, v in category_scores.items()}

        n = sum(p.values())
        p = {k: v / n for k, v in p.items()}
        return p

    def _category_p(self, category: str) -> float:
        return self._category_count[category] / sum(self._category_count.values())

    def _calc_text_score(self, category: str, word_counts: dict) -> int:
        score = 0
        for word in word_counts:
            # Word hit
            if word in self._calculated_word_scores[category]:
                word_score = self._calculated_word_scores[category][word]
            else:
                word_score = self._calculated_word_scores[category][None]

            score += word_score * word_counts[word]

        return score

    def _get_word_count(self, category: str, word: str) -> float:
        if word in self._category_words[category]:
            return float(self._category_words[category][word])
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

        is_spam = spam_rate > 0.99

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
    obj = NaiveBayes()
    obj.fit("マック", [
        "ハンバーガー ポテト ジュース",
        "ハンバーガー チキン ジュース",
        "ハンバーガー",
        "ポテト チキン",
    ])
    obj.fit("ケンタッキー", [
        "チキン ポテト ジュース",
        "ハンバーガー ポテト チキン",
        "ジュース チキン チキン",
        "ハンバーガー チキン チキン ポテト ビスケット ジュース"
    ])

    print(obj.classify("ポテト ハンバーガー ジュース"))
    print(obj.classify("チキン"))
