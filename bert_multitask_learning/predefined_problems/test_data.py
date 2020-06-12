

import re
import string
import random


from .ner_data import gold_horse_ent_type_process_fn, read_ner_data, gold_horse_segment_process_fn
from ..preproc_decorator import preprocessing_fn


def get_weibo_fake_cls_fn(file_path):
    @preprocessing_fn
    def weibo_fake_cls(params, mode):
        """Just a test problem to test multiproblem support

        Arguments:
            params {Params} -- params
            mode {mode} -- mode
        """
        data = read_ner_data(file_pattern=file_path,
                             proc_fn=gold_horse_ent_type_process_fn)
        if mode == 'train':
            data = data['train']
        else:
            data = data['eval']
        inputs_list = data['inputs']
        target_list = data['target']

        new_target_list = ['1' if len(
            set(t)) > 1 else '0' for t in target_list]

        return inputs_list, new_target_list
    return weibo_fake_cls


def get_weibo_fake_seq2seq_tag_fn(file_path):
    @preprocessing_fn
    def weibo_fake_seq2seq_tag(params, mode: str):

        data = read_ner_data(file_pattern=file_path,
                             proc_fn=gold_horse_ent_type_process_fn)
        if mode == 'train':
            data = data['train']
        else:
            data = data['eval']
        inputs_list = data['inputs']
        target_list = data['target']
        new_target_list = [['1', '2'] for _ in range(len(inputs_list))]
        return inputs_list, new_target_list
    return weibo_fake_seq2seq_tag


def get_weibo_pretrain_fn(file_path):
    @preprocessing_fn
    def weibo_pretrain(params, mode):

        sentence_split = r'[.!?。？！]'

        data = read_ner_data(file_pattern=file_path,
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

        return segmented_list
    return weibo_pretrain


def get_weibo_fake_seq_tag_fn(file_path):
    @preprocessing_fn
    def weibo_fake_seq_tag(params, mode):
        data = read_ner_data(file_pattern=file_path,
                             proc_fn=gold_horse_ent_type_process_fn)
        if mode == 'train':
            data = data['train']
        else:
            data = data['eval']
        inputs_list = data['inputs']
        target_list = data['target']

        return inputs_list, target_list
    return weibo_fake_seq_tag


def get_weibo_fake_multi_cls_fn(file_path):
    @preprocessing_fn
    def weibo_fake_multi_cls(params, mode):
        data = read_ner_data(file_pattern=file_path,
                             proc_fn=gold_horse_ent_type_process_fn)
        if mode == 'train':
            data = data['train']
        else:
            data = data['eval']
        inputs_list = data['inputs']

        # create fake target
        target_list = []
        for _ in inputs_list:
            target_list.append(
                list(string.ascii_lowercase[:random.randint(0, 25)]))

        return inputs_list, target_list
    return weibo_fake_multi_cls
