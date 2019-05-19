

import re

from ..tokenization import FullTokenizer

from ..utils import (
    get_or_make_label_encoder, BOS_TOKEN, EOS_TOKEN)
from ..create_generators import create_pretraining_generator, create_single_problem_generator

from .ner_data import gold_horse_ent_type_process_fn, read_ner_data


def weibo_fake_cls(params, mode):
    """Just a test problem to test multiproblem support

    Arguments:
        params {Params} -- params
        mode {mode} -- mode
    """
    tokenizer = FullTokenizer(vocab_file=params.vocab_file)
    data = read_ner_data(file_pattern='data/ner/weiboNER*',
                         proc_fn=gold_horse_ent_type_process_fn)
    if mode == 'train':
        data = data['train']
    else:
        data = data['eval']
    inputs_list = data['inputs'][:100]
    target_list = data['target'][:100]

    new_target_list = ['1' if len(set(t)) > 1 else '0' for t in target_list]

    label_encoder = get_or_make_label_encoder(
        params, 'weibo_fake_cls', mode, new_target_list, '0')

    return create_single_problem_generator('weibo_fake_cls',
                                           inputs_list,
                                           new_target_list,
                                           label_encoder,
                                           params,
                                           tokenizer,
                                           mode)


def weibo_fake_seq2seq_tag(params, mode: str):

    tokenizer = FullTokenizer(vocab_file=params.vocab_file)
    data = read_ner_data(file_pattern='data/ner/weiboNER*',
                         proc_fn=gold_horse_ent_type_process_fn)
    if mode == 'train':
        data = data['train']
    else:
        data = data['eval']
    inputs_list = data['inputs'][:100]
    target_list = data['target'][:100]
    new_target_list = [['1', '2'] for t in target_list]
    label_encoder = get_or_make_label_encoder(
        params,
        'weibo_fake_seq2seq_tag',
        mode,
        [BOS_TOKEN, '1', '2', EOS_TOKEN],
        zero_class=BOS_TOKEN)

    return create_single_problem_generator(
        'weibo_fake_seq2seq_tag',
        inputs_list,
        new_target_list,
        label_encoder,
        params,
        tokenizer,
        mode)


def weibo_pretrain(params, mode):

    sentence_split = r'[.!?。？！]'

    tokenizer = FullTokenizer(vocab_file=params.vocab_file)
    data = read_ner_data(file_pattern='data/ner/weiboNER*',
                         proc_fn=gold_horse_segment_process_fn)
    if mode == 'train':
        data = data['train']
    else:
        data = data['eval']
    inputs_list = data['inputs']

    segmented_list = []
    for document in inputs_list:
        segmented_list.append([])
        doc_string = ''.join(document)
        splited_doc = re.split(sentence_split, doc_string)
        for sentence in splited_doc:
            if sentence:
                segmented_list[-1].append(list(sentence))
    segmented_list = [doc for doc in segmented_list if doc]

    return create_pretraining_generator('weibo_pretrain',
                                        segmented_list,
                                        None,
                                        None,
                                        params,
                                        tokenizer,
                                        mode)


def weibo_fake_seq_tag(params, mode):
    tokenizer = FullTokenizer(vocab_file=params.vocab_file)
    data = read_ner_data(file_pattern='data/ner/weiboNER*',
                         proc_fn=gold_horse_ent_type_process_fn)
    if mode == 'train':
        data = data['train']
    else:
        data = data['eval']
    inputs_list = data['inputs'][:100]
    target_list = data['target'][:100]

    flat_label = [item for sublist in target_list for item in sublist]

    label_encoder = get_or_make_label_encoder(
        params, 'weibo_fake_seq_tag', mode, flat_label)

    return create_single_problem_generator('weibo_fake_seq_tag',
                                           inputs_list,
                                           target_list,
                                           label_encoder,
                                           params,
                                           tokenizer,
                                           mode)
