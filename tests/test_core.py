from fasttext_lite import FastTextClassifier
import pytest
import numpy as np


@pytest.fixture
def sentences():
    sentences = [
        "The cat sat on the mat.",
        "I woke up early this morning.",
        "The sun is shining brightly today.",
        "I like to eat pizza for dinner.",
        "She wore a yellow dress to the party.",
        "He plays basketball every weekend.",
        "The book is on the table.",
        "I am going to the grocery store.",
        "The sky is blue.",
        "My favorite color is green.",
        "He drank a glass of water.",
        "She brushed her teeth before bed.",
        "The movie was really good.",
        "The dog barked loudly.",
        "I need to do laundry today.",
        "He rode his bike to the park.",
        "She took a nap in the afternoon.",
        "The flower smells nice.",
        "He wrote a letter to his friend.",
        "I am learning to speak Spanish.",
        "She loves to dance in her free time.",
        "The car drove down the street.",
        "He listened to music while he worked.",
        "She cooked dinner for her family.",
        "The tree is tall and green.",
        "I enjoy reading books on my Kindle.",
        "He went to bed early last night.",
        "She ran a marathon last year.",
        "The water in the pool is cold.",
        "I watched a movie on Netflix.",
        "He took his dog for a walk.",
        "She bought a new shirt at the mall.",
        "The train arrived on time.",
        "I need to buy some new shoes.",
        "He played the guitar at the concert.",
        "She went to the beach over the weekend.",
        "The computer is running slowly.",
        "I love to eat ice cream on hot days.",
        "He works at a bank downtown.",
        "She went to the doctor for a check-up.",
        "The bird flew away when we got too close.",
        "I need to get my oil changed soon.",
        "He went on vacation to Hawaii.",
        "She likes to drink coffee in the morning.",
        "The soccer game ended in a tie.",
        "I need to clean my room.",
        "He went to the library to study.",
        "She wrote a poem about the ocean.",
        "The snow is melting in the sun.",
        "I enjoy watching movies with my friends.",
        "He went to the museum to see the art exhibit.",
        "She sang a song in the shower.",
        "The clock struck twelve.",
        "I drank a cup of coffee this morning.",
        "He watched TV all day.",
        "The rain is pouring outside.",
        "I went to the gym this afternoon.",
        "She baked cookies for her friends.",
        "The pencil is on the desk.",
        "He went for a run in the park.",
        "The fish in the aquarium are swimming around.",
        "I need to call my mom later.",
        "She bought a new phone yesterday.",
        "The moon is full tonight.",
        "He took a nap on the couch.",
        "The clouds are blocking the sun.",
        "I like to read books before bed.",
        "She went to the store to buy groceries.",
        "The bird chirped in the tree.",
        "He cooked dinner for his family.",
        "The grass is green and soft.",
        "I need to take my car to the mechanic.",
        "She danced with her friends at the club.",
        "The car drove past the stop sign.",
        "He listened to a podcast on his way to work.",
        "The water in the lake is calm.",
        "I like to watch sports on TV.",
        "She went for a walk in the park.",
        "The flower pot is on the windowsill.",
        "He wrote a shopping list for the supermarket.",
        "The airplane flew high in the sky.",
        "I ate a sandwich for lunch.",
        "She watched a movie at home.",
        "The bookshelf is full of books.",
        "He went to the gym to lift weights.",
        "The trees are turning colors in the fall.",
        "I need to buy a new pair of headphones.",
        "She likes to go hiking in the mountains.",
        "The sun is setting in the west.",
        "He went to the beach to surf.",
        "The coffee shop is crowded in the morning.",
        "I like to listen to music while I work.",
        "She went to the concert with her friends.",
        "The boat sailed across the ocean.",
        "He played a game of chess with his friend.",
        "The baby cried for his mother.",
        "I went to the park to play basketball.",
        "She watched the sunset over the ocean.",
        "The train station is busy in the morning.",
        "He went to the store to buy a birthday present.",
        "The bridge crosses over the river.",
        "I need to get my hair cut soon.",
        "She likes to paint in her spare time.",
        "The football game was on TV.",
    ]
    return sentences


def test_fit_predict(sentences):
    X = sentences
    y = ["Label 1" if i % 2 == 0 else "Label 2" for i in range(len(X))]
    clf = FastTextClassifier()
    clf.fit(X, y)
    clf.predict(["This is a sentence to predict on"])
    clf.predict_proba(["This is a sentence to predict on", "this is another sentence"])


def test_save_load(sentences, tmp_path):
    X = sentences
    y = ["Label 1" if i % 2 == 0 else "Label 2" for i in range(len(X))]
    clf = FastTextClassifier()
    clf.fit(X, y)
    X_test = [
        "This is a sentence to predict on",
        "This is another sentence to predict on",
    ]
    p = clf.predict(X_test)
    pp = clf.predict_proba(X_test)
    clf.save(str(tmp_path))
    clf2 = FastTextClassifier.load(str(tmp_path))
    p2 = clf2.predict(X_test)
    pp2 = clf2.predict_proba(X_test)
    assert p == p2
    assert np.allclose(pp, pp2)


def test_save_load_quantized(sentences, tmp_path):
    X = sentences
    y = ["Label 1" if i % 2 == 0 else "Label 2" for i in range(len(X))]
    clf = FastTextClassifier()
    clf.fit(X, y)
    X_test = [
        "This is a sentence to predict on",
        "This is another sentence to predict on",
    ]
    p = clf.predict(X_test)
    pp = clf.predict_proba(X_test)
    clf.save(str(tmp_path), quantized=True)
    clf2 = FastTextClassifier.load(str(tmp_path))
    p2 = clf2.predict(X_test)
    pp2 = clf2.predict_proba(X_test)
    assert p == p2
    assert np.allclose(pp, pp2)
    assert clf.classes_ == clf2.classes_
    assert clf.adjusted_labels == clf2.adjusted_labels


def test_fallback(sentences, tmp_path):
    X = sentences
    y = ["Label 1" if i % 2 == 0 else "Label 2" for i in range(len(X))]
    clf = FastTextClassifier()
    clf.fit(X, y)
    clf.save(str(tmp_path))
    import fasttext

    model = fasttext.load_model(str(tmp_path / "fasttext.bin"))
    model.predict(["This is a sentence to predict on"])


def test_adjusted_labels(sentences):
    X = sentences
    y = ["FIRST LABEL" if i % 2 == 0 else "SECOND LABEL" for i in range(len(X))]
    clf = FastTextClassifier()
    clf.fit(X, y)
    assert clf.adjusted_labels == {
        "FIRST LABEL": "FIRST_LABEL",
        "SECOND LABEL": "SECOND_LABEL",
    }
