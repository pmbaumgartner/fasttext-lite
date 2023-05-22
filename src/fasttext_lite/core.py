from itertools import chain
import json
from pathlib import Path
import fasttext
from typing import Dict, List, Union

from tempfile import NamedTemporaryFile
from typing import Literal
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError


def _adjust_label(string: str) -> str:
    return string.replace(" ", "_")


def _check_fit(model):
    if not model.fitted:
        raise NotFittedError(
            "This FastTextClassifier instance is not fitted yet."
            " Call 'fit' with appropriate arguments before using this estimator."
        )


StrOrPath = Union[str, Path]


def convert_path(path: StrOrPath) -> Path:
    if isinstance(path, Path):
        return path
    elif isinstance(path, str):
        return Path(path)
    else:
        raise TypeError("Not `str` or `Path`")


class FastTextClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        lr: float = 0.1,
        dim: int = 100,
        ws: int = 5,
        epoch: int = 5,
        minCount: int = 1,
        minCountLabel: int = 1,
        minn: int = 0,
        maxn: int = 0,
        neg: int = 5,
        wordNgrams: int = 1,
        loss: Literal["ns", "hs", "softmax", "ova"] = "softmax",
        bucket: int = 2000000,
        lrUpdateRate: int = 100,
        t: float = 0.0001,
        label: str = "__label__",
        verbose: int = 2,
        thread: int = 2,
    ):
        self.lr = lr
        self.dim = dim
        self.ws = ws
        self.epoch = epoch
        self.minCount = minCount
        self.minCountLabel = minCountLabel
        self.minn = minn
        self.maxn = maxn
        self.neg = neg
        self.wordNgrams = wordNgrams
        self.loss = loss
        self.bucket = bucket
        self.lrUpdateRate = lrUpdateRate
        self.t = t
        self.label = label
        self.verbose = verbose
        self.thread = thread
        # non-model
        self.fitted = False
        self.is_quantized = False
        self.adjusted_labels: Dict[str, str] = {}

    def fit(self, X, y) -> None:
        self.original_labels = self.sort_labels(y)
        self.adjusted_labels = {
            label: _adjust_label(label) for label in self.original_labels
        }
        self.adjusted_labels_inverse = {v: k for k, v in self.adjusted_labels.items()}
        with NamedTemporaryFile() as train_file:
            with open(train_file.name, "a") as f:
                for text, label in zip(X, y):
                    f.write(f"__label__{self.adjusted_labels[label]} {text}\n")

            self.model = fasttext.train_supervised(
                input=train_file.name,
                lr=self.lr,
                dim=self.dim,
                ws=self.ws,
                epoch=self.epoch,
                minCount=self.minCount,
                minCountLabel=self.minCountLabel,
                minn=self.minn,
                maxn=self.maxn,
                neg=self.neg,
                wordNgrams=self.wordNgrams,
                loss=self.loss,
                bucket=self.bucket,
                lrUpdateRate=self.lrUpdateRate,
                t=self.t,
                label=self.label,
                verbose=self.verbose,
                thread=self.thread,
            )
        self.classes_ = self.original_labels
        self.fitted = True

    @property
    def n_labels(self):
        _check_fit(self)
        return len(self.classes_)

    def predict(self, X):
        _check_fit(self)
        if isinstance(X, str):
            X = [X]
        preds = self.model.predict(X, k=1)
        # Remove the label prefix from the string (e.g. "__label__")
        labels = [labels[0][len(self.label) :] for labels in preds[0]]

        return labels

    def predict_proba(self, X):
        # Note: This will be slow because we have to re-sort :(
        # NOTE: This has some sorting error when using cross_val_predict
        _check_fit(self)
        if isinstance(X, str):
            X = [X]
        preds = self.model.predict(X, k=self.n_labels)
        preds_array = np.zeros(shape=(len(X), self.n_labels), dtype=np.float32)
        for p_i, (labels, probs) in enumerate(zip(*preds)):
            # The labels are returned in descending order of probs,
            # we want them consistently ordered by our alphabetic classes
            label_order = {
                label: i
                for i, label in enumerate(label[len(self.label) :] for label in labels)
            }
            for l_i, ordered_label in enumerate(self.classes_):
                preds_array[p_i, l_i] = probs[
                    label_order[self.adjusted_labels[ordered_label]]
                ]
        return preds_array

    @classmethod
    def sort_labels(cls, y: List[str]) -> List[str]:
        return list(sorted(list(set(y))))

    def save(self, path: StrOrPath, quantized=False) -> None:
        path = convert_path(path)
        path.mkdir(parents=True, exist_ok=True)
        """Save to a directory"""
        if quantized:
            self.model.quantize()
        extension = "ftz" if quantized else "bin"
        params = {
            "lr": self.lr,
            "dim": self.dim,
            "ws": self.ws,
            "epoch": self.epoch,
            "minCount": self.minCount,
            "minCountLabel": self.minCountLabel,
            "minn": self.minn,
            "maxn": self.maxn,
            "neg": self.neg,
            "wordNgrams": self.wordNgrams,
            "loss": self.loss,
            "bucket": self.bucket,
            "lrUpdateRate": self.lrUpdateRate,
            "t": self.t,
        }
        (path / "params.json").write_text(json.dumps(params, indent=4))
        labels_data = {
            "labels": [
                {"original": label, "adjusted": self.adjusted_labels[label]}
                for label in self.original_labels
            ]
        }
        (path / "labels.json").write_text(json.dumps(labels_data, indent=4))

        self.model.save_model(str(path / f"fasttext.{extension}"))

    @classmethod
    def load(cls, path: StrOrPath) -> "FastTextClassifier":
        path = convert_path(path)
        params = json.loads((path / "params.json").read_text())
        labels_data = json.loads((path / "labels.json").read_text())
        classes_ = []
        adjusted_labels = {}
        for label in labels_data["labels"]:
            classes_.append(label["original"])
            adjusted_labels[label["original"]] = label["adjusted"]
        try:
            model_file = next(
                chain(path.glob("fasttext.ftz"), path.glob("fasttext.bin"))
            )
        except StopIteration:
            raise ValueError("No file with .bin or .ftz extension in directory.")
        clf = cls(**params)
        clf.model = fasttext.load_model(str(model_file))
        clf.fitted = True
        clf.is_quantized = clf.model.is_quantized()
        clf.classes_ = classes_
        clf.adjusted_labels = adjusted_labels
        clf.adjusted_labels_inverse = {v: k for k, v in adjusted_labels.items()}

        return clf

    def _get_original_label(self, adjusted_label: str):
        return self.adjusted_labels_inverse[adjusted_label]

    def _get_original_label_index(self, adjusted_label: str):
        return self.original_labels.index(self._get_original_label(adjusted_label))
