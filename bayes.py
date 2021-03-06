import numpy as np
import re
import unicodedata
import MeCab  # https://pypi.org/project/mecab-python3/
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# sudo apt-get install mecab mecab-ipadic-utf8
# -d /var/lib/mecab/dic/ipadic-utf8
wakati = MeCab.Tagger("-Owakati -d /var/lib/mecab/dic/ipadic-utf8")
vec_tfidf = TfidfVectorizer()


class Vectorizer(object):
    def __init__(self):
        self._vocabularies = set()
        self.vec_tfidf = TfidfVectorizer()

    def fit(self, texts: list):
        self._vocabularies = set()

        for text in texts:
            words = set(self.to_words(text))
            self._vocabularies |= words

    def vectorize_text(self, text: str):
        words = self.to_words(text)
        counts = self.count_words(words)
        counts = {k: v for k, v in counts.items() if k in self._vocabularies}

        vector = {w: 0 for w in self._vocabularies}
        for k, v in counts.items():
            vector[k] += v

        return [v for v in vector.values()]

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
        t = text
        t = unicodedata.normalize("NFKC", t)
        t = re.sub(r'[\s,.、。()]+', r' ', t)
        t = re.sub(r'[0-9]', r'0', text)
        t = t.lower()
        return t

    def _remove_dc_words(self, words: list) -> list:
        discard_words = r"\s+"
        words = [w for w in words if not re.match(discard_words, w)]
        return words

    def _separate_text(self, text: str) -> list:
        return wakati.parse(text).split()


class TfidfVec(Vectorizer):
    def __init__(self):
        self.vec = TfidfVectorizer()

    def fit(self, texts):
        self.vec.fit([" ".join(self.to_words(text)) for text in texts])

    def vectorize_text(self, text):
        t = " ".join(self.to_words(text))
        return self.vec.transform([t]).toarray()[0]


class CountVec(Vectorizer):
    def __init__(self):
        self.vec = CountVectorizer()

    def fit(self, texts):
        self.vec.fit([" ".join(self.to_words(text)) for text in texts])

    def vectorize_text(self, text):
        t = " ".join(self.to_words(text))
        return self.vec.transform([t]).toarray()[0]


class NaiveBayes(Vectorizer):
    def __init__(self):
        # ラプラススムージングのパラメーター
        self.smoothing_a = 1
        self.smoothing_b = 2

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
        self._calculated_word_weight = {}

    def fit(self, category: str, texts: list):
        if category in self._category_count:
            raise Exception("%s is already registered" % category)

        # キャッシュを初期化
        self._calculated_word_weight = None
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

    def pre_calculate_word_weight(self):
        self._calculated_word_weight = {}

        for category in self._category_count:
            evidence = float(
                sum(self._category_words[category].values())
                + len(self._vocabularies) * self.smoothing_b
            )
            self._calculated_word_weight[category] = {}

            for word in self._category_words[category]:
                word_score = np.log10((
                    self._get_word_count(category, word) +
                    self.smoothing_a
                ) / evidence)

                self._calculated_word_weight[category][word] = word_score

            nohit_score = np.log10(self.smoothing_a / evidence)
            self._calculated_word_weight[category][None] = nohit_score

    def classify(self, text: str) -> dict:
        if self._calculated_word_weight is None:
            self.pre_calculate_word_weight()

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
            if word in self._calculated_word_weight[category]:
                word_score = self._calculated_word_weight[category][word]
            else:
                word_score = self._calculated_word_weight[category][None]

            score += word_score * word_counts[word]

        return score

    def _get_word_count(self, category: str, word: str) -> float:
        if word in self._category_words[category]:
            return float(self._category_words[category][word])
        return 0.0


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

    # {'マック': 0.7531448106571573, 'ケンタッキー': 0.2468551893428426}
    print(obj.classify("ポテト ハンバーガー ジュース"))

    # {'マック': 0.009035030522802552, 'ケンタッキー': 0.9909649694771974}
    print(obj.classify("チキン ビスケット"))
