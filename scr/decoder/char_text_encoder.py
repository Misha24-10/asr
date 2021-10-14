import re


class Alphabet():
    def __init__(self):
        self.dictin = 'а, б, в, г, д, е, ж, з, и, й, к, л, м, н, о, п, р, с, т, у, ф, х, ц, ч, ш, щ, ъ, ы, ь, э, ю, я, ^,  '.split(
            ', ')
        self.all_cahracters = set(self.dictin)
        self.char_int = {char: index for index, char in enumerate(self.dictin)}
        self.int_char = {v: k for k, v in self.char_int.items()}

    def char2int(self, sentence):
        sentence = sentence.replace('ё', 'е')
        sentence = sentence.lower()
        sentence = re.sub(r"[^а-я ]", "", sentence)
        list_of_int_characters = []
        for char in sentence:
            list_of_int_characters.append(self.char_int[char])
        return list_of_int_characters

    def int2char(self, list_of_ints):
        sentence = ""
        for i in list(list_of_ints):
            sentence += self.int_char[i]
        return sentence
