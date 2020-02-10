import os
import random
import tempfile
from copy import copy
from functools import partial
from multiprocessing import cpu_count
from multiprocessing import Pool
import pickle

import numpy as np
import tensorflow as tf
from joblib import Parallel, delayed

from .utils import BOS_TOKEN, EOS_TOKEN
from .bert_preprocessing.create_bert_features import create_bert_features


def _float_list_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_list_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_fn(features: dict):
    features_tuple = {}
    for feature_name, feature in features.items():
        if type(feature) is list:
            feature = np.array(feature)
        if type(feature) is np.ndarray:
            if issubclass(feature.dtype.type, np.integer):
                features_tuple[feature_name] = _int64_list_feature(
                    feature.flatten())
            elif issubclass(feature.dtype.type, np.float):
                features_tuple[feature_name] = _float_list_feature(
                    feature.flatten())

            features_tuple['{}_shape'.format(
                feature_name)] = _int64_list_feature(feature.shape)

        elif type(feature) is float:
            features_tuple[feature_name] = _float_feature(feature)
            features_tuple['{}_shape'.format(
                feature_name)] = _int64_list_feature([])
        else:
            features_tuple[feature_name] = _int64_feature(feature)
            features_tuple['{}_shape'.format(
                feature_name)] = _int64_list_feature([])

    example_proto = tf.train.Example(
        features=tf.train.Features(feature=features_tuple))
    return example_proto.SerializeToString()


def make_tfrecord(data_list, output_dir, serialize_fn, mode='train', shards_per_file=10000, prefix='', **kwargs):
    """Function to make TF Records from csv files
    This function will take all csv files in data_dir, convert them
    to tf example and write to *_{suffix}_train/eval.tfrecord to data_dir.
    Arguments:
        data_dir {str} -- dir that has csv files and store tf record
        generator_fn {fn} -- A function that takes a list of filepath and yield the
        parsed recored from file.
        serialize_fn {fn} -- A function that takes output of generator fn and convert to tf example
    Keyword Arguments:
        suffix {str} -- suffix to add to tf record files (default: {''})
    """

    # create output tfrecord path
    file_list = [os.path.join(
        output_dir, prefix, '{}_{:05d}.tfrecord'.format(mode, i)) for i in range(len(data_list))]

    os.makedirs(os.path.dirname(file_list[0]), exist_ok=True)

    def _write_fn(d_list, path, serialize_fn):
        print('Writing {}'.format(os.path.basename(path)))
        with tf.io.TFRecordWriter(path) as writer:
            for features in d_list:
                example = serialize_fn(features)
                writer.write(example)

    _write_part_fn = partial(_write_fn, serialize_fn=serialize_fn)

    Parallel(min(cpu_count(), len(file_list)))(delayed(_write_part_fn)(d_list=x, path=y)
                                               for x, y in zip(data_list, file_list))


def write_single_problem_tfrecord(problem,
                                  inputs_list,
                                  target_list,
                                  label_encoder,
                                  params,
                                  tokenizer,
                                  mode):
    """Function to create iterator for single problem

    This function will:
        1. Do some text cleaning using original bert tokenizer, if
            problem type is sequential tagging, corresponding labels
            will be removed.

            Example:
                Before: inputs: ['a', '&', 'c'] target: [0, 0, 1]
                After: inputs: ['a', 'c'] target: [0, 1]
        2. Add [CLS], [SEP] tokens
        3. Padding
        4. yield result dict

    Arguments:
        problem {str} -- problem name
        inputs_list {list } -- inputs list
        target_list {list} -- target list, should have the same length as inputs list
        label_encoder {LabelEncoder} -- label encoder
        params {Params} -- params
        tokenizer {tokenizer} -- Bert Tokenizer
        epoch {int} -- Deprecate
    """

    problem_type = params.problem_type[problem]

    # whether this problem is sequential labeling
    # for sequential labeling, targets needs to align with any
    # change of inputs
    is_seq = problem_type in ['seq_tag']

    example_list = list(zip(inputs_list, target_list))
    # split data_list by shards_per_file
    data_shards = []
    for i in range(0, len(example_list), 5000):
        data_shards.append(example_list[i:i + 5000])

    part_fn = partial(create_bert_features, problem=problem,
                      label_encoder=label_encoder,
                      params=params,
                      tokenizer=tokenizer,
                      mode=mode,
                      problem_type=problem_type,
                      is_seq=is_seq)
    data_list = Parallel(min(cpu_count(), len(data_shards)))(delayed(part_fn)(example_list=d_list)
                                                             for d_list in data_shards)

    make_tfrecord(data_list=data_list,
                  output_dir=params.tmp_file_dir, serialize_fn=serialize_fn)


def create_generator(params, mode):
    """Function to create iterator for multiple problem

    This function dose the following things:
    1. Create dummy labels for each problems.
    2. Initialize all generators
    3. Sample a problem to train at this batch. If eval, take turns
    4. Create a loss multiplier
    5. Tried to generate samples for target problem, if failed, init gen
    6. Add dummy label to other problems

    Example:
        Problem: cws|NER|weibo_ner&weibo_cws
        1. Dummy labels: cws_label_ids: [0,0,0] ...
        2. Blablabla
        3. Sample, say (weibo_ner&weibo_cws)
        4. loss multipliers: {'cws_loss_multiplier': 0, ..., 'weibo_ner_loss_multiplier': 1, ...}
        ...

    Arguments:
        params {Params} -- params
        mode {mode} -- mode
    """
    # example
    # problem_list: ['NER', 'cws', 'weibo_ner', 'weibo_cws']
    # problem_chunk: [['NER'], ['cws'], ['weibo_ner', 'weibo_cws']]
    problem_list = []
    problem_chunk = []
    for problem_dict in params.run_problem_list:
        problem_list += list(problem_dict.keys())
        problem_chunk.append(list(problem_dict.keys()))

    # get dummy labels
    def _create_dummpy_label(problem_type, num_classes=None):
        if problem_type == 'cls':
            return 0
        elif problem_type == 'seq_tag':
            return [0]*params.max_seq_len
        elif problem_type == 'mask_lm':
            return [0]*params.max_predictions_per_seq
        elif problem_type == 'multi_cls':
            return [0]*num_classes
        else:
            return [0]*params.decode_max_seq_len
    dummy_label_dict = {problem+'_label_ids': _create_dummpy_label(
        params.problem_type[problem], params.num_classes[problem]) for problem in problem_list if params.problem_type[problem] != 'pretrain'}
    dummy_label_dict.update({problem+'_mask': _create_dummpy_label(
        params.problem_type[problem]) for problem in problem_list if params.problem_type[problem] in ['seq2seq_tag', 'seq2seq_text']})

    pretrain_dummpy_dict = {
        "masked_lm_positions": _create_dummpy_label('mask_lm'),
        "masked_lm_ids": _create_dummpy_label('mask_lm'),
        "masked_lm_weights": _create_dummpy_label('mask_lm'),
        "next_sentence_label_ids": _create_dummpy_label('cls'),
        "next_sentence_loss_multiplier": 0,
        "masked_lm_loss_multiplier": 0}

    # init gen
    try:
        gen_dict = {problem: params.read_data_fn[problem](params, mode)
                    for problem in problem_list}
    except KeyError:
        not_exist_problem = [
            p for p in problem_list if p not in params.read_data_fn]
        raise KeyError(
            'The preprocessing function of {0} not found, make sure you called params.add_problem. If you\'re using train_bert_multitask, please make sure you provided problem_type_dict and processing_fn_dict.'.format(not_exist_problem))

    while gen_dict:
        # sample problem to train
        if len(problem_chunk) > 1:
            data_num_list = [params.data_num_dict[chunk[0]]
                             for chunk in problem_chunk]
            if params.multitask_balance_type == 'data_balanced':
                sample_prob = np.array(data_num_list) / np.sum(data_num_list)
                current_problem_chunk_ind = np.random.choice(
                    list(range(len(problem_chunk))), p=sample_prob)
                current_problem_chunk = problem_chunk[current_problem_chunk_ind]

            elif params.multitask_balance_type == 'problem_balanced':
                sample_prob = np.array(
                    [1]*len(data_num_list)) / np.sum([1]*len(data_num_list))
                current_problem_chunk_ind = np.random.choice(
                    list(range(len(problem_chunk))), p=sample_prob)
                current_problem_chunk = problem_chunk[current_problem_chunk_ind]
        else:
            current_problem_chunk = problem_chunk[0]

        # create loss multiplier
        loss_multiplier = {
            problem+'_loss_multiplier': 0 for problem in problem_list if params.problem_type[problem] != 'pretrain'}
        for problem in current_problem_chunk:
            if params.problem_type[problem] != 'pretrain':
                loss_multiplier[problem+'_loss_multiplier'] = 1
            else:
                loss_multiplier['next_sentence_loss_multiplier'] = 1
                loss_multiplier['masked_lm_loss_multiplier'] = 1

        base_dict = {}
        base_input = None
        for problem in current_problem_chunk:
            try:
                instance = next(gen_dict[problem])
            except StopIteration:
                if mode == 'train':
                    gen_dict[problem] = params.read_data_fn[problem](
                        params, mode)
                    instance = next(gen_dict[problem])
                else:
                    del gen_dict[problem]
                    continue
            except KeyError:
                continue

            if not instance:
                continue

            base_dict.update(instance)
            if base_input is None:
                base_input = instance['input_ids']
            elif not params.augument_mask_lm:
                assert base_input == instance[
                    'input_ids'], 'Inputs id of two chained problem not aligned. Please double check!'

        if not base_dict:
            continue

        for problem in problem_list:
            problem_type = params.problem_type[problem]
            problem_label_name = '{0}_label_ids'.format(problem)

            if problem_label_name not in base_dict:
                if problem_type == 'seq_tag':
                    base_dict[problem_label_name] = dummy_label_dict[problem_label_name][:len(
                        base_dict['input_ids'])]
                elif problem_type == 'pretrain':
                    if 'masked_lm_ids' not in base_dict:
                        base_dict.update(pretrain_dummpy_dict)
                else:
                    base_dict[problem_label_name] = dummy_label_dict[problem_label_name]

                if problem_type in ['seq2seq_tag', 'seq2seq_text']:
                    base_dict['{0}_mask'.format(
                        problem)] = dummy_label_dict['{0}_mask'.format(problem)]

        # add loss multipliers
        base_dict.update(loss_multiplier)
        yield base_dict
