import spacy
import numpy as np
import pickle

nlp = spacy.load("en_core_web_lg")

def cosine(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def key_to_lem(key):
    return preprocess_word(nlp.vocab[key].text)


def preprocess_word(word):
    return nlp(word.lower())[0].lemma_


def get_vector(word):
    return nlp(preprocess_word(word))[0].vector


def similar_to_word(word, n):
    token = nlp(preprocess_word(word))[0]
    tokvec = np.asarray([token.vector])
    most_similar = nlp.vocab.vectors.most_similar(tokvec, n=n)

    return most_similar


def most_similar_excluding(vec, words_to_exclude, n):
    most_similar = nlp.vocab.vectors.most_similar(np.asarray([vec]), n=5*n)

    words_to_exclude_clean = {preprocess_word(w) for w in words_to_exclude}

    results = []
    results_set = set()

    for (key, best_row, score) in zip(
        most_similar[0][0], most_similar[1][0], most_similar[2][0]):
        lem = key_to_lem(key)
        if lem not in words_to_exclude_clean and lem not in results_set:
            results_set.add(lem)
            results.append((lem, score))

    return results[:n]


def most_similar_excluding_word(word, words_to_exclude, n):
    token = nlp(preprocess_word(word))[0]

    return most_similar_excluding(token.vector, words_to_exclude + [word], n)


try:
    with open("./nouns.pickle", "rb") as f:
        nouns = pickle.load(f)
except FileNotFoundError:
    print("nouns.pickle not found: regenerating")
    with open("./english-nouns.txt") as f:
        nouns = []
        for noun in f:
            noun_lem = preprocess_word(noun.strip())
            nouns.append((noun_lem, get_vector(noun_lem)))

    with open("./nouns.pickle", "wb") as f:
        pickle.dump(nouns, f)
