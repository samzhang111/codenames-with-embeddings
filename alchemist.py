from shared import most_similar_excluding, get_vector


class AlchemistGame(object):
    def __init__(self, init_words=['fire', 'earth', 'air', 'water']):
        self.words = set(init_words)

    def combine_words(self, words):
        vecs = [get_vector(word) for word in words]

        avg_vec = sum(vecs)/len(vecs)

        try:
            new_word, score = most_similar_excluding(
                avg_vec, words + list(self.words), n=10
            )[0]
        except IndexError:
            print("No word found!")
            return

        ingredients_list = ' + '.join(words)
        print(f"{ingredients_list} = {new_word}! (similarity={score:.2f})")

        if new_word not in self.words:
            self.words.add(new_word)

        return new_word

    def game_loop(self):
        while True:
            print(f"Your words are {self.words}")

            missing_ingredient = True

            while missing_ingredient:
                raw_ingredients = input("Enter words (separated by spaces): ")
                ingredients = raw_ingredients.split()

                if len(ingredients) > 0:

                    i = 0
                    for ingredient in ingredients:
                        if ingredient not in self.words:
                            print(f"{ingredient} not in words, try again")
                            missing_ingredient = True
                        else:
                            i += 1

                    if i == len(ingredients):
                        missing_ingredient = False
                        print("")
                        self.combine_words(ingredients)


alchemist = AlchemistGame()
alchemist.game_loop()
