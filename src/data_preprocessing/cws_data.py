import sys
import os
import glob
from tqdm import tqdm

from bert.tokenization import FullTokenizer

from ..utils import (get_or_make_label_encoder,
                     create_single_problem_generator)


def process_line_msr_pku(l):
    decoded_line = l.strip().split('  ')
    return [w.strip('\r\n') for w in decoded_line]


def process_line_as_training(l):
    decoded_line = l.strip().split('\u3000')
    return [w.strip('\r\n') for w in decoded_line]


def process_line_cityu(l):
    decoded_line = l.strip().split(' ')
    return [w.strip('\r\n') for w in decoded_line]


def get_process_fn(filename):

    if 'msr' in filename or 'pk' in filename:
        return process_line_msr_pku

    elif 'as' in filename:
        return process_line_as_training

    elif 'cityu' in filename:
        return process_line_cityu


def _process_text_files(path_list):

    # Create possible tags for fast lookup
    possible_tags = []
    for i in range(1, 300):
        if i == 1:
            possible_tags.append('s')
        else:
            possible_tags.append('b' + 'm' * (i - 2) + 'e')

    inputs = []
    target = []

    for s in range(len(path_list)):
        filename = path_list[s]

        # Init left and right queue

        with open(filename, 'r', encoding='utf8') as f:

            input_list = f.readlines()

            process_fn = get_process_fn(os.path.split(filename)[-1])

            for l in tqdm(input_list):
                pos_tag = []
                final_line = []

                decoded_line = process_fn(l)

                for w in decoded_line:
                    if w and len(w) <= 299:
                        final_line.append(w)
                        pos_tag.append(possible_tags[len(w) - 1])

                decode_str = ''.join(final_line)

                pos_tag_str = ''.join(pos_tag)

                if len(pos_tag_str) != len(decode_str):
                    print('Skip one row. ' + pos_tag_str + ';' + decode_str)
                    continue

                inputs.append(list(decode_str))
                target.append(list(pos_tag_str))

    return inputs, target


def CWS(params, mode):

    tokenizer = FullTokenizer(vocab_file=params.vocab_file)
    if mode == 'train':
        file_list = glob.glob('data/cws/training/*.utf8')
    else:
        file_list = ['as_testing_gold.utf8',
                     'cityu_test_gold.utf8', 'msr_test_gold.utf8', 'pku_test_gold.utf8']
        # file_list = ['msr_test_gold.utf8']
        file_list = [os.path.join('data/cws/gold', f) for f in file_list]

    inputs, target = _process_text_files(file_list)

    label_encoder = get_or_make_label_encoder(
        'CWS', mode, ['b', 'm', 'e', 's'], zero_class='[PAD]')

    return create_single_problem_generator('CWS',
                                           inputs,
                                           target,
                                           label_encoder,
                                           params,
                                           tokenizer)
