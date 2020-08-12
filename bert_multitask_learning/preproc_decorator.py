import os
from sklearn.preprocessing import MultiLabelBinarizer
from types import GeneratorType
import logging

from .utils import get_or_make_label_encoder, cluster_alphnum, LabelEncoder
from .special_tokens import TRAIN, EVAL, PREDICT
from .read_write_tfrecord import write_single_problem_chunk_tfrecord, write_single_problem_gen_tfrecord
from .bert_preprocessing.tokenization import FullTokenizer


def preprocessing_fn(func):
    def wrapper(params, mode, get_data_num=False, write_tfrecord=True):
        problem = func.__name__

        tokenizer = FullTokenizer(
            vocab_file=params.vocab_file, do_lower_case=True)
        example_list = func(params, mode)

        if isinstance(example_list, GeneratorType):
            if get_data_num:
                # create label encoder and data num
                cnt = 0
                label_list = []
                logging.info(
                    "Preprocessing function returns generator, might take some time to create label encoder...")
                for example in example_list:
                    cnt += 1
                    try:
                        _, label = example
                        label_list.append(label)
                    except ValueError:
                        pass

                # create label encoder
                label_encoder = get_or_make_label_encoder(
                    params, problem=problem, mode=mode, label_list=label_list)

                if label_encoder is None:
                    return cnt, 0
                elif isinstance(label_encoder, LabelEncoder):
                    return cnt, len(label_encoder.encode_dict)
                elif isinstance(label_encoder, MultiLabelBinarizer):
                    return cnt, label_encoder.classes_.shape[0]
                else:
                    # label_encoder is tokenizer
                    return cnt, len(label_encoder.vocab)
            else:
                # create label encoder
                label_encoder = get_or_make_label_encoder(
                    params, problem=problem, mode=mode, label_list=[])

            if mode == PREDICT:
                return example_list, label_encoder

            if write_tfrecord:
                return write_single_problem_gen_tfrecord(
                    func.__name__,
                    example_list,
                    label_encoder,
                    params,
                    tokenizer,
                    mode)
            else:
                return {
                    'problem': func.__name__,
                    'gen': example_list,
                    'label_encoder': label_encoder,
                    'tokenizer': tokenizer
                }

        else:
            try:
                inputs_list, target_list = example_list
            except ValueError:
                inputs_list = example_list
                target_list = None

            label_encoder = get_or_make_label_encoder(
                params, problem=problem, mode=mode, label_list=target_list)

            if get_data_num:
                if label_encoder is None:
                    return len(inputs_list), 0
                elif isinstance(label_encoder, LabelEncoder):
                    return len(inputs_list), len(label_encoder.encode_dict)
                elif isinstance(label_encoder, MultiLabelBinarizer):
                    return len(inputs_list), label_encoder.classes_.shape[0]
                else:
                    # label_encoder is tokenizer
                    return len(inputs_list), len(label_encoder.vocab)

            if mode == PREDICT:
                return inputs_list, target_list, label_encoder

            if write_tfrecord:
                return write_single_problem_chunk_tfrecord(
                    func.__name__,
                    inputs_list,
                    target_list,
                    label_encoder,
                    params,
                    tokenizer,
                    mode)
            else:
                return {
                    'problem': func.__name__,
                    'inputs_list': inputs_list,
                    'target_list': target_list,
                    'label_encoder': label_encoder,
                    'tokenizer': tokenizer
                }

    return wrapper
