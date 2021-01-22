# naive-bayes
ただの単純ベイズ分類器です。

## 依存関係
```console
sudo apt install mecab libmecab-dev mecab-ipadic-utf8
pip install mecab-python3 numpy 
```

# 実行方法


```python
from bayes import NaiveBayes

obj = NaiveBayes()

# 学習 
# obj.fit("ラベル", [doc1, doc2, ...])
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

# 推論
print(obj.classify("ポテト ハンバーガー ジュース"))  # {'マック': 0.7531448106571573, 'ケンタッキー': 0.2468551893428426}
print(obj.classify("チキン ビスケット"))  # {'マック': 0.009035030522802552, 'ケンタッキー': 0.9909649694771974}
```
