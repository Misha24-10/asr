import editdistance
from scr.decoder.char_text_encoder import Alphabet
from scr.decoder.ctc_decode import ctc_decode

def calc_wer(target_text: str, pred_text: str):
    targ_split = target_text.split()
    pred_split = pred_text.split()
    if len(targ_split) == 0 and len(pred_split) != 0:
        return 1.0
    if len(targ_split) == 0 and len(pred_split) == 0:
        return 0
    error = editdistance.eval(targ_split, pred_split)
    return error / len(targ_split)


def calc_cer(target_text: str, pred_text: str):
    if (len(target_text) == 0 and len(pred_text) != 0):
        return 1.0
    if len(target_text) == 0 and len(pred_text) == 0:
        return 0
    error = editdistance.eval(target_text, pred_text)
    return error / len(target_text)


def calculate_cer(targets, decodings, y_lengths, x_lengths):
    """
    Calculate the Levenshtein distance between predictions and GT
    """
    numper_exmapes = 3
    alph = Alphabet()
    cer = 0.0
    wer = 0.0
    targets = targets.detach().tolist()

    decodings = decodings.detach().tolist()



    for i, target in enumerate(targets):
        targets[i] = target[:y_lengths[i]]

    for i, pred in enumerate(decodings):
        decodings[i] = pred[:x_lengths[i]]

    pairs = ""
    for target, d in zip(targets, decodings):
        target = alph.int2char(target)
        decoding = ctc_decode(d)

        cer += calc_cer(target, decoding)
        wer += calc_wer(target, decoding)

    for tar, pred in zip(targets,decodings):
        print("\nTrue: ",alph.int2char(tar),"\n","Predic: ",ctc_decode(pred))
        numper_exmapes = numper_exmapes - 1
        if (numper_exmapes == 0):
            break
    return cer / len(targets), wer / len(targets), pairs
