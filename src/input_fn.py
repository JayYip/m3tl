from collections import defaultdict
from tqdm import tqdm
import numpy as np

import tensorflow as tf

from .tokenization import FullTokenizer

from .params import Params
from .utils import (create_generator, tokenize_text_with_seqs, truncate_seq_pair,
                    add_special_tokens_with_seqs, create_mask_and_padding)


def train_eval_input_fn(config: Params, mode='train', epoch=None):

    def gen():
        if mode == 'train':
            epoch = config.train_epoch
        else:
            epoch = 1

        g = create_generator(params=config, mode=mode, epoch=epoch)
        for example in g:
            yield example

    output_type = {
        'input_ids': tf.int32,
        'input_mask': tf.int32,
        'segment_ids': tf.int32
    }
    output_shapes = {
        'input_ids': [config.max_seq_len],
        'input_mask': [config.max_seq_len],
        'segment_ids': [config.max_seq_len]
    }
    if config.augument_mask_lm:
        output_type.update({
            "masked_lm_positions": tf.int32,
            "masked_lm_ids": tf.int32,
            "masked_lm_weights": tf.float32
        })

        output_shapes.update({
            "masked_lm_positions": [config.max_predictions_per_seq],
            "masked_lm_ids": [config.max_predictions_per_seq],
            "masked_lm_weights": [config.max_predictions_per_seq]
        })
    for problem_dict in config.run_problem_list:
        for problem, problem_type in problem_dict.items():
            output_type.update({'%s_loss_multiplier' % problem: tf.int32})
            output_shapes.update({'%s_loss_multiplier' % problem: []})

            if problem_type in ['seq_tag']:
                output_type.update({'%s_label_ids' % problem: tf.int32})
                output_shapes.update(
                    {'%s_label_ids' % problem: [config.max_seq_len]})
            elif problem_type in ['cls']:
                output_type.update({'%s_label_ids' % problem: tf.int32})
                output_shapes.update({'%s_label_ids' % problem: []})
            elif problem_type in ['seq2seq_tag', 'seq2seq_text']:
                output_type.update({'%s_label_ids' % problem: tf.int32})
                output_shapes.update(
                    {'%s_label_ids' % problem: [config.decode_max_seq_len]})

                output_type.update({'%s_mask' % problem: tf.int32})
                output_shapes.update(
                    {'%s_mask' % problem: [config.decode_max_seq_len]})

            elif problem_type in ['pretrain']:
                output_type.update({
                    "masked_lm_positions": tf.int32,
                    "masked_lm_ids": tf.int32,
                    "masked_lm_weights": tf.float32,
                    "next_sentence_label_ids": tf.int32
                })

                output_shapes.update({
                    "masked_lm_positions": [config.max_predictions_per_seq],
                    "masked_lm_ids": [config.max_predictions_per_seq],
                    "masked_lm_weights": [config.max_predictions_per_seq],
                    "next_sentence_label_ids": []
                })

    tf.logging.info(output_type)
    tf.logging.info(output_shapes)

    dataset = tf.data.Dataset.from_generator(
        gen, output_types=output_type, output_shapes=output_shapes)

    if mode == 'train':
        dataset = dataset.shuffle(config.shuffle_buffer)

    dataset = dataset.prefetch(config.prefetch)
    if mode == 'train':
        dataset = dataset.batch(config.batch_size)
    else:
        dataset = dataset.batch(config.batch_size*2)
    return dataset


def predict_input_fn(input_file_or_list, config: Params, mode='predict'):

    # if is string, treat it as path to file
    if isinstance(input_file_or_list, str):
        inputs = open(input_file_or_list, 'r', encoding='utf8').readlines()
    else:
        inputs = input_file_or_list

    tokenizer = FullTokenizer(config.vocab_file)

    # data_dict = {}
    # data_dict['input_ids'] = []
    # data_dict['input_mask'] = []
    # data_dict['segment_ids'] = []

    def gen():
        data_dict = {}
        for doc in tqdm(inputs, desc='Processing Inputs'):
            inputs_a = list(doc)
            tokens, target = tokenize_text_with_seqs(
                tokenizer, inputs_a, None)

            tokens_a, tokens_b, target = truncate_seq_pair(
                tokens, None, target, config.max_seq_len)

            tokens, segment_ids, target = add_special_tokens_with_seqs(
                tokens_a, tokens_b, target)

            input_mask, tokens, segment_ids, target = create_mask_and_padding(
                tokens, segment_ids, target, config.max_seq_len)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            data_dict['input_ids'] = input_ids
            data_dict['input_mask'] = input_mask
            data_dict['segment_ids'] = segment_ids
            yield data_dict
    output_type = {
        'input_ids': tf.int32,
        'input_mask': tf.int32,
        'segment_ids': tf.int32
    }
    output_shapes = {
        'input_ids': [config.max_seq_len],
        'input_mask': [config.max_seq_len],
        'segment_ids': [config.max_seq_len]
    }
    # dataset = tf.data.Dataset.from_tensor_slices(data_dict)
    dataset = tf.data.Dataset.from_generator(
        gen, output_types=output_type, output_shapes=output_shapes)
    dataset = dataset.batch(config.batch_size*2)

    return dataset


def to_serving_input(input_file_or_list, config: Params, mode='predict', tokenizer=None):
        # if is string, treat it as path to file
    if isinstance(input_file_or_list, str):
        inputs = open(input_file_or_list, 'r', encoding='utf8').readlines()
    else:
        inputs = input_file_or_list

    data_dict = {}
    for doc in inputs:
        inputs_a = list(doc)
        tokens, target = tokenize_text_with_seqs(
            tokenizer, inputs_a, None)

        tokens_a, tokens_b, target = truncate_seq_pair(
            tokens, None, target, config.max_seq_len)

        tokens, segment_ids, target = add_special_tokens_with_seqs(
            tokens_a, tokens_b, target)

        input_mask, tokens, segment_ids, target = create_mask_and_padding(
            tokens, segment_ids, target, config.max_seq_len)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        data_dict['input_ids'] = input_ids
        data_dict['input_mask'] = input_mask
        data_dict['segment_ids'] = segment_ids
        for k in data_dict:
            data_dict[k] = np.expand_dims(data_dict[k], axis=0)
        yield data_dict


def serving_input_fn():
    features = {
        'input_ids': tf.placeholder(tf.int32, [None, None]),
        'input_mask': tf.placeholder(tf.int32, [None, None]),
        'segment_ids': tf.placeholder(tf.int32, [None, None])
    }
    return tf.estimator.export.ServingInputReceiver(features, features)
