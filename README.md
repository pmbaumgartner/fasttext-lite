# FastText Lite

A scikit-learn API version of FastText.

## Example

```python
from fasttext_lite import FastTextClassifier

X = ["This is a sentence", "This is another sentence"]
y = ["Label 1", "Label 2"]
clf = FastTextClassifier(epoch=25, wordNgrams=2, minCount=2)
clf.fit(X, y)
p = clf.predict(["This is a sentence to predict on"])

clf.save("mymodel")  # saves to directory.

new_clf = FastTextClassifier.load("mymodel")

# The save method calls the original fasttext save option
# and saves to a file `fasttext.bin` (or `fasttext.ftz` if quantized)
# so you can always drop back to that library

import fasttext

model = fasttext.load_model("mymodel/fasttext.bin")
model.predict(["This is another sentence to predict on"])
```