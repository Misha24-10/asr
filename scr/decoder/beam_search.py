from fast_ctc_decode import beam_search, viterbi_search
from scr.metrics.metrics import *
def beam_serch_eval(batch):
    batch = batch.exp().to('cpu').detach().numpy()
    alphabet = "^абвгдежзийклмнопрстуфхцчшщъыьэюя "
    pred_list = [(beam_search(element.T, alphabet, beam_size=50, beam_cut_threshold=10**(-30)))[0] for element in batch]
    return pred_list
