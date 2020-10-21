import logging
from types import GeneratorType
from typing import Callable

from sklearn.preprocessing import MultiLabelBinarizer

from .read_write_tfrecord import (write_single_problem_chunk_tfrecord,
                                  write_single_problem_gen_tfrecord)
from .special_tokens import PREDICT
from .utils import LabelEncoder, get_or_make_label_encoder, load_transformer_tokenizer


def preprocessing_fn(func: Callable):
    """Usually used as a decorator.

    The input and output signature of decorated function should be:
    func(params: bert_multitask_learning.BaseParams,
         mode: str) -> Union[Generator[X, y], Tuple[List[X], List[y]]]

    Where X can be:
    - Dicitionary of 'a' and 'b' texts: {'a': 'a test', 'b': 'b test'}
    - Text: 'a test'
    - Dicitionary of modalities: {'text': 'a test', 'image': np.array([1,2,3])}

    Where y can be:
    - Text or scalar: 'label_a'
    - List of text or scalar: ['label_a', 'label_a1'] (for seq2seq and seq_tag)

    This decorator will do the following things:
    - load tokenizer
    - call func, save as example_list
    - create label_encoder and count the number of rows of example_list
    - create bert features from example_list and write tfrecord

    Args:
        func (Callable): preprocessing function for problem
    """
    def wrapper(params, mode, get_data_num=False, write_tfrecord=True):
        problem = func.__name__

        tokenizer = load_transformer_tokenizer(
            params.transformer_tokenizer_name, params.transformer_tokenizer_loading)
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
                if isinstance(label_encoder, LabelEncoder):
                    return cnt, len(label_encoder.encode_dict)
                if isinstance(label_encoder, MultiLabelBinarizer):
                    return cnt, label_encoder.classes_.shape[0]

                # label_encoder is tokenizer
                try:
                    return cnt, len(label_encoder.vocab)
                except AttributeError:
                    # models like xlnet's vocab size can only be retrieved from config instead of tokenizer
                    return cnt, params.bert_decoder_config.vocab_size

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
                if isinstance(label_encoder, LabelEncoder):
                    return len(inputs_list), len(label_encoder.encode_dict)
                if isinstance(label_encoder, MultiLabelBinarizer):
                    return len(inputs_list), label_encoder.classes_.shape[0]
                if hasattr(label_encoder, 'vocab'):
                    # label_encoder is tokenizer
                    return len(inputs_list), len(label_encoder.vocab)
                elif hasattr(params, 'decoder_vocab_size'):
                    return len(inputs_list), params.decoder_vocab_size
                else:
                    raise ValueError('Cannot determine num of classes for problem {0}.'
                                     'This is usually caused by {1} dose not has attribute vocab. In this case, you should manually specify vocab size to params: params.decoder_vocab_size = 32000'.format(problem, type(label_encoder).__name__))

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
