from scr.decoder.char_text_encoder import Alphabet

def ctc_decode(inds):
    """
    Decode hypotheses
        1) Remove repetitive letters
        2) Remove blank characters
    """
    alph = Alphabet()
    text = alph.int2char(inds)
    processed_text = ""
    for i, value in enumerate(text):
        if (i > 0 and value != processed_text[-1]):
            processed_text += value
        if (i == 0 and value != ''):
            processed_text += value
    processed_text = processed_text.replace('^', '')  # delete empty tokens from strings
    if processed_text == "":
        return ""
    if processed_text[0] == " ":
        processed_text = processed_text[1:]
    return processed_text
