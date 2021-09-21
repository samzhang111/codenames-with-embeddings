from shared import cosine, most_similar_excluding, get_vector


class AlchemistGame(object):
    def __init__(self, init_words=['fire', 'earth', 'air', 'water']):
        self.words = set(init_words)


    def combine_words(self, word1, word2):
        if word1 not in self.words:
            raise ValueError(f"{word1} not in words")

        if word2 not in self.words:
            raise ValueError(f"{word2} not in words")

        vec1 = get_vector(word1)
        vec2 = get_vector(word2)

        avg_vec = (vec1 + vec2)/2

        try:
            new_word = most_similar_excluding(avg_vec, [word1, word2], n=10)[0][0]
        except IndexError:
            print("No word found!")
            return

        print(f"{word1} + {word2} = {new_word}!")

        if new_word not in self.words:
            self.words.add(new_word)

        return new_word
