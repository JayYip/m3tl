import sys
import os
import glob
from tqdm import tqdm
import re

from sklearn.model_selection import train_test_split

from ..utils import filter_empty
from .preproc_decorator import preprocessing_fn


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

                decoded_line = process_fn(l)

                last_isalpha_num = False
                pos_tag = []
                final_line = []

                for w in decoded_line:
                    is_alphanum = bool(re.match('^[a-zA-Z0-9]+$', w))

                    if not is_alphanum:
                        # if this round is chinese and last round is alphanum,
                        # append the last alphanum's target: s and continue
                        if last_isalpha_num:
                            pos_tag.append('s')

                        if w and len(w) <= 299:
                            final_line.extend(list(w))
                            pos_tag.extend(list(possible_tags[len(w) - 1]))
                        last_isalpha_num = False
                    else:
                        if last_isalpha_num:
                            final_line[-1] += w
                        else:
                            final_line.append(w)
                            last_isalpha_num = True

                if last_isalpha_num:
                    pos_tag.append('s')

                # decode_str = ''.join(final_line)

                # pos_tag_str = ''.join(pos_tag)

                if len(pos_tag) != len(final_line):
                    print(filename)
                    print('Skip one row. ' + pos_tag + ';' + final_line)
                    continue

                inputs.append(final_line)
                target.append(pos_tag)

    return inputs, target


@preprocessing_fn
def cws(params, mode):
    file_list = glob.glob('data/ctb8.0/data/segmented/*')

    input_list = []
    target_list = []

    # Create possible tags for fast lookup
    possible_tags = []
    for i in range(1, 300):
        if i == 1:
            possible_tags.append('s')
        else:
            possible_tags.append('b' + 'm' * (i - 2) + 'e')

    for file_path in file_list:
        with open(file_path, 'r', encoding='utf8') as f:
            raw_doc_list = f.readlines()
        text_row_ind = [i+1 for i,
                        text in enumerate(raw_doc_list) if '<S ID=' in text]

        sentence_list = [text for i,
                         text in enumerate(raw_doc_list) if i in text_row_ind]

        for sentence in sentence_list:
            input_list.append([])
            target_list.append([])
            for word in sentence.split():
                if word and len(word) <= 299:
                    tag = possible_tags[len(word) - 1]
                    input_list[-1] += list(word)
                    target_list[-1] += list(tag)
                else:
                    continue

    if mode == 'train':
        input_list, _, target_list, _ = train_test_split(
            input_list, target_list, test_size=0.2, random_state=3721)
    else:
        _, input_list, _, target_list = train_test_split(
            input_list, target_list, test_size=0.2, random_state=3721)

    if mode == 'train':
        file_list = [  # 'as_testing_gold.utf8',
            'cityu_training.utf8', 'msr_training.utf8', 'pku_training.utf8']
        file_list = [os.path.join('data/cws/training', f) for f in file_list]
    else:
        file_list = [  # 'as_testing_gold.utf8',
            'cityu_test_gold.utf8', 'msr_test_gold.utf8', 'pku_test_gold.utf8']
        # file_list = ['msr_test_gold.utf8']
        file_list = [os.path.join('data/cws/gold', f) for f in file_list]

    icwb_inputs, icwb_target = _process_text_files(file_list)

    input_list += icwb_inputs
    target_list += icwb_target

    input_list, target_list = filter_empty(input_list, target_list)

    return input_list, target_list


@preprocessing_fn
def as_cws(params, mode):

    if mode == 'train':
        file_list = glob.glob('data/cws/training/as_*.utf8')
    else:
        file_list = ['as_testing_gold.utf8']
        # file_list = ['msr_test_gold.utf8']
        file_list = [os.path.join('data/cws/gold', f) for f in file_list]

    input_list, target_list = _process_text_files(file_list)

    return input_list, target_list


@preprocessing_fn
def msr_cws(params, mode):

    if mode == 'train':
        file_list = glob.glob('data/cws/training/msr_*.utf8')
    else:
        file_list = ['msr_test_gold.utf8']
        # file_list = ['msr_test_gold.utf8']
        file_list = [os.path.join('data/cws/gold', f) for f in file_list]

    input_list, target_list = _process_text_files(file_list)

    return input_list, target_list


@preprocessing_fn
def pku_cws(params, mode):

    if mode == 'train':
        file_list = glob.glob('data/cws/training/pku_*.utf8')
    else:
        file_list = ['pku_test_gold.utf8']
        # file_list = ['msr_test_gold.utf8']
        file_list = [os.path.join('data/cws/gold', f) for f in file_list]

    input_list, target_list = _process_text_files(file_list)

    return input_list, target_list


@preprocessing_fn
def city_cws(params, mode):

    if mode == 'train':
        file_list = glob.glob('data/cws/training/cityu_*.utf8')
    else:
        file_list = ['cityu_test_gold.utf8']
        # file_list = ['msr_test_gold.utf8']
        file_list = [os.path.join('data/cws/gold', f) for f in file_list]

    input_list, target_list = _process_text_files(file_list)

    return input_list, target_list
