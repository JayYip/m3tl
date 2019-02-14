import glob
import os
import sys

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from ..tokenization import FullTokenizer

from ..utils import create_single_problem_generator, get_or_make_label_encoder
from .ctb_data import read_ctbpos


def POS(params, mode):
    tokenizer = FullTokenizer(vocab_file=params.vocab_file)

    input_list, target_list = read_ctbpos()

    if mode == 'train':
        input_list, _, target_list, _ = train_test_split(
            input_list, target_list, test_size=0.2, random_state=3721)
    else:
        _, input_list, _, target_list = train_test_split(
            input_list, target_list, test_size=0.2, random_state=3721)

    flat_target_list = [item for sublist in target_list for item in sublist]

    label_encoder = get_or_make_label_encoder(
        params, 'POS', mode, flat_target_list, zero_class='[PAD]')
    return create_single_problem_generator('POS',
                                           input_list,
                                           target_list,
                                           label_encoder,
                                           params,
                                           tokenizer,
                                           mode)
