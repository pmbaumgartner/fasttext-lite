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

## Multilabel example
```python
import numpy as np
from fasttext_lite import FastTextMultiOutputClassifier

X = ["Pad thai", "Ice cream", "Tamarindo"]
classes = ["Spicy", "Sweet", "Sour"]
Y = np.array([
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1]
])
clf = FastTextMultiOutputClassifier(epoch=25, wordNgrams=2, minCount=2)
clf.fit(X, Y, classes)
p = clf.predict_proba(["Thai iced tea"])
```