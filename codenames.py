from collections import defaultdict

import numpy as np

from shared import nlp, cosine, most_similar_excluding


class CodenameGuesser(object):
    def __init__(self, words):
        self.words = words
        self.toks = [nlp(word)[0] for word in words]


    def make_move(self, clue, num_guesses):
        clue_tok = nlp(clue)[0]

        guesses = []
        for n in range(num_guesses):
            best_score = 0
            best_tok = None

            for tok in self.toks:
                score = cosine(tok.vector, clue_tok.vector)
                if score > best_score:
                    best_score = score
                    best_tok = tok

            guesses.append(best_tok)
            self.toks = [x for x in self.toks if x != best_tok]

        return guesses


def construct_similarity_matrix(positive_toks, negative_toks):
    pos_pos_matrix = defaultdict(list)
    pos_neg_matrix = defaultdict(list)
    for p in positive_toks:
        for p2 in positive_toks:
            if str(p) == str(p2):
                continue

            score = cosine(p.vector, p2.vector)
            pos_pos_matrix[str(p)].append((p2, score))

        for n in negative_toks:
            score = cosine(p.vector, n.vector)
            pos_neg_matrix[str(p)].append((n, score))

    return pos_pos_matrix, pos_neg_matrix


class CodenameSpymaster(object):
    def __init__(self, positive_words, negative_words):
        self.positive_words = positive_words
        self.positive_toks = nlp(" ".join(positive_words))
        self.negative_words = negative_words
        self.negative_toks = nlp(" ".join(negative_words))
        self.pos_pos_matrix, self.pos_neg_matrix = construct_similarity_matrix(
            self.positive_toks, self.negative_toks
        )


    def compute_subscore(self, guess_vector, word_set, rho_vec):
        rho = cosine(guess_vector, rho_vec)
        hardest_guess = min([cosine(guess_vector, w.vector)  for w in word_set])

        return hardest_guess - rho


    def find_move(self):
        best_score = 0
        best_subscore = 0
        best_set = []
        guess = None

        for pos_tok in self.positive_toks:
            rho_word, rho = max(
                self.pos_neg_matrix[str(pos_tok)], key=lambda x: x[1])
            rho_vec = rho_word.vector

            closer_than_any_negative_toks = [pos_tok]
            for p2, p2_score in self.pos_pos_matrix[str(pos_tok)]:
                if p2_score > rho:
                    closer_than_any_negative_toks.append(p2)

            if len(closer_than_any_negative_toks) == best_score:
                centroid = np.asarray([x.vector for x in closer_than_any_negative_toks]).mean(axis=0)
                guess = nlp(most_similar_excluding(centroid,
                    self.positive_words + self.negative_words,
                    10)[0][0])[0]
                subscore = self.compute_subscore(guess.vector,
                                                      closer_than_any_negative_toks,
                                                      rho_vec)

                if subscore > best_subscore:
                    best_subscore = subscore
                    best_set = closer_than_any_negative_toks

            if len(closer_than_any_negative_toks) > best_score:
                best_score = len(closer_than_any_negative_toks)
                best_set = closer_than_any_negative_toks
                centroid = np.asarray([x.vector for x in best_set]).mean(axis=0)
                guess = nlp(most_similar_excluding(centroid,
                    self.positive_words + self.negative_words,
                    10)[0][0])[0]
                best_subscore = self.compute_subscore(guess.vector,
                                                      best_set,
                                                      rho_vec)

        return (guess, best_score)


pos_words = ['oasis', 'gangster', 'head', 'lap', 'fighter', 'platypus',
             'hotel', 'button']

neg_words = ['heart', 'london', 'wall', 'bread', 'potato', 'paper', 'soldier',
             'memory', 'thief']

spy = CodenameSpymaster(pos_words, neg_words)
guesser = CodenameGuesser(pos_words + neg_words)
