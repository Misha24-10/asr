from fast_ctc_decode import beam_search, viterbi_search
from scr.metrics.metrics import *
def beam_serch_eval(batch):
    batch = batch.exp().to('cpu').detach().numpy()
    alphabet = "^абвгдежзийклмнопрстуфхцчшщъыьэюя "
    pred_list = [(beam_search(element.T, alphabet, beam_size=50, beam_cut_threshold=10**(-30)))[0] for element in batch]
    return pred_list
def calculate_cer_beam(targets,decodings,y_lengths):
    """
    Calculate the Levenshtein distance between predictions and GT
    """
    numper_exmapes = 2
    alph = Alphabet()
    cer = 0.0
    wer = 0.0
    targets = targets.detach().tolist()
    for i, target in enumerate(targets):
        targets[i] = target[:y_lengths[i]]
    pairs = ""
    for target, d in zip(targets, decodings):
        target = alph.int2char(target)
        cer += calc_cer(target, d)
        wer += calc_wer(target, d)

    for tar, pred in zip(targets,decodings):
        pairs = "True: "+ alph.int2char(tar) +" -- "+"Predic: " + pred
        break
    return cer / len(targets), wer / len(targets), pairs