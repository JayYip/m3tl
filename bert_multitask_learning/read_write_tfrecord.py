import os
import random
import tempfile
from copy import copy
from functools import partial
from multiprocessing import cpu_count
import pickle
import json
from glob import glob
import logging

import numpy as np
import tensorflow as tf
from joblib import Parallel, delayed

from .special_tokens import BOS_TOKEN, EOS_TOKEN, TRAIN, EVAL
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


def serialize_fn(features: dict, return_feature_desc=False):
    features_tuple = {}
    feature_desc = {}
    for feature_name, feature in features.items():
        if type(feature) is list:
            feature = np.array(feature)
        if type(feature) is np.ndarray:
            if issubclass(feature.dtype.type, np.integer):
                features_tuple[feature_name] = _int64_list_feature(
                    feature.flatten())
                feature_desc[feature_name] = 'int64'
            elif issubclass(feature.dtype.type, np.float):
                features_tuple[feature_name] = _float_list_feature(
                    feature.flatten())
                feature_desc[feature_name] = 'float32'

            features_tuple['{}_shape'.format(
                feature_name)] = _int64_list_feature(feature.shape)
            feature_desc['{}_shape_value'.format(feature_name)] = feature.shape

        elif type(feature) is float:
            features_tuple[feature_name] = _float_feature(feature)
            features_tuple['{}_shape'.format(
                feature_name)] = _int64_list_feature([])
            feature_desc[feature_name] = 'float32'
            feature_desc['{}_shape_value'.format(feature_name)] = feature.shape
        else:
            features_tuple[feature_name] = _int64_feature(feature)
            features_tuple['{}_shape'.format(
                feature_name)] = _int64_list_feature([])
            feature_desc[feature_name] = 'int64'
            feature_desc['{}_shape_value'.format(feature_name)] = feature.shape

        feature_desc['{}_shape'.format(
            feature_name)] = 'int64'
        feature_desc['{}_shape_value'.format(feature_name)] = [
            None for _ in feature.shape]

    example_proto = tf.train.Example(
        features=tf.train.Features(feature=features_tuple))
    if return_feature_desc:
        return example_proto.SerializeToString(), feature_desc

    return example_proto.SerializeToString()


def make_tfrecord(data_list, output_dir, serialize_fn, mode='train', shards_per_file=10000, prefix='', **kwargs):

    # create output tfrecord path
    file_list = [os.path.join(
        output_dir, prefix, '{}_{:05d}.tfrecord'.format(mode, i)) for i in range(len(data_list))]

    os.makedirs(os.path.dirname(file_list[0]), exist_ok=True)

    def _write_fn(d_list, path, serialize_fn, mode='train'):
        logging.info('Writing {}'.format(path))
        feature_desc_path = os.path.join(os.path.dirname(
            path), '{}_feature_desc.json'.format(mode))

        return_list = []
        with tf.io.TFRecordWriter(path) as writer:
            for features in d_list:
                example, feature_desc = serialize_fn(
                    features, return_feature_desc=True)
                writer.write(example)
                if not os.path.exists(feature_desc_path):
                    json.dump(feature_desc, open(
                        feature_desc_path, 'w', encoding='utf8'))

    _write_part_fn = partial(_write_fn, serialize_fn=serialize_fn, mode=mode)

    for x, y in zip(data_list, file_list):
        _write_part_fn(d_list=x, path=y)


def write_single_problem_tfrecord(problem,
                                  inputs_list,
                                  target_list,
                                  label_encoder,
                                  params,
                                  tokenizer,
                                  mode):

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
                  output_dir=params.tmp_file_dir,
                  serialize_fn=serialize_fn,
                  prefix=problem,
                  mode=mode)


def write_tfrecord(params, replace=False):
    """Write TFRecord for every problem

    Output location: params.tmp_file_dir

    Arguments:
        params {params} -- params

    Keyword Arguments:
        replace {bool} -- Whether to replace existing tfrecord (default: {False})
    """

    read_data_fn_dict = params.read_data_fn
    path_list = []
    for problem in params.problem_list:
        read_fn = read_data_fn_dict[problem]
        file_dir = os.path.join(params.tmp_file_dir, problem)
        if not os.path.exists(file_dir) or replace:
            read_fn(params, TRAIN)
            read_fn(params, EVAL)


def make_feature_desc(feature_desc_dict: dict):
    feature_desc = {}
    for feature_name, feature_type in feature_desc_dict.items():
        if feature_type == 'int64':
            feature_desc[feature_name] = tf.io.VarLenFeature(tf.int64)
        elif feature_type == 'float32':
            feature_desc[feature_name] = tf.io.VarLenFeature(tf.float32)

    return feature_desc


def reshape_tensors_in_dataset(example):
    """Reshape serialized tensor back to its original shape

    Arguments:
        example {Example} -- Example

    Returns:
        Example -- Example
    """

    for feature_key in example:
        example[feature_key] = tf.sparse.to_dense(example[feature_key])

    for feature_key in example:
        if '_shape' in feature_key:
            continue

        shape_tensor = example['{}_shape'.format(feature_key)]
        example[feature_key] = tf.reshape(example[feature_key], shape_tensor)

    for feature_key in list(example.keys()):
        if '_shape' in feature_key:
            del example[feature_key]

    return example


def add_loss_multiplier(example, problem):
    example['{}_loss_multiplier'.format(problem)] = tf.constant(
        value=1, shape=(), dtype=tf.int32)
    return example


def set_shape_for_dataset(example, feature_desc_dict):
    for feature_key in example:
        example[feature_key].set_shape(
            feature_desc_dict['{}_shape_value'.format(feature_key)])
    return example


def get_dummy_features(dataset_dict, feature_desc_dict):
    """Get dummy features.
    Dummy features are used to make sure every feature dict
    at every iteration has the same keys.

    Example:
        problem A: {'input_ids': [1,2,3], 'A_label_ids': 1}
        problem B: {'input_ids': [1,2,3], 'B_label_ids': 2}

    Then dummy features:
        {'A_label_ids': 0, 'B_label_ids': 0}

    At each iteration, we sample a problem, let's say we sampled A
    Then:
        feature dict without dummy:
            {'input_ids': [1,2,3], 'A_label_ids': 1}
        feature dict with dummy:
            {'input_ids': [1,2,3], 'A_label_ids': 1, 'B_label_ids':0}

    Arguments:
        dataset_dict {dict} -- dict of datasets of all problems

    Returns:
        dummy_features -- dict of dummy tensors
    """

    feature_keys = [list(d.output_shapes.keys())
                    for _, d in dataset_dict.items()]
    common_features_accross_problems = set(
        feature_keys[0]).intersection(*feature_keys[1:])

    dummy_features = {}
    for problem, problem_dataset in dataset_dict.items():
        output_types = problem_dataset.output_types
        dummy_features.update({k: tf.cast(tf.constant(shape=[1 for _ in feature_desc_dict.get('{}_shape_value'.format(k), [])], value=0), v)
                               for k, v in output_types.items()
                               if k not in common_features_accross_problems})
    return dummy_features


def add_dummy_features_to_dataset(example, dummy_features):
    """Add dummy features to dataset

    feature dict without dummy:
        {'input_ids': [1,2,3], 'A_label_ids': 1}
    feature dict with dummy:
        {'input_ids': [1,2,3], 'A_label_ids': 1, 'B_label_ids':0}

    Arguments:
        example {data example} -- dataset example
        dummy_features {dict} -- dict of dummy tensors
    """
    for feature_name in dummy_features:
        if feature_name not in example:
            example[feature_name] = tf.identity(dummy_features[feature_name])
    return example


def read_tfrecord(params, mode: str):
    """Read and parse TFRecord for every problem

    The returned dataset is parsed, reshaped, to_dense tensors
    with dummy features.

    Arguments:
        params {params} -- params
        mode {str} -- mode, train, eval or predict

    Returns:
        dict -- dict with keys: problem name, values: dataset
    """
    dataset_dict = {}
    all_feature_desc_dict = {}
    for problem in params.problem_list:
        file_dir = os.path.join(params.tmp_file_dir, problem)
        tfrecord_path_list = glob(os.path.join(
            file_dir, '{}_*.tfrecord'.format(mode)))
        feature_desc_dict = json.load(
            open(os.path.join(file_dir, '{}_feature_desc.json'.format(mode))))
        all_feature_desc_dict.update(feature_desc_dict)
        feature_desc = make_feature_desc(feature_desc_dict)
        dataset = tf.data.TFRecordDataset(tfrecord_path_list)
        dataset = dataset.map(lambda x: tf.io.parse_single_example(
            x, feature_desc)).map(reshape_tensors_in_dataset).map(
                lambda x: set_shape_for_dataset(x, feature_desc_dict)
        ).map(lambda x: add_loss_multiplier(x, problem))
        dataset_dict[problem] = dataset

    # add dummy features
    dummy_features = get_dummy_features(dataset_dict, all_feature_desc_dict)
    for problem in params.problem_list:
        dataset_dict[problem] = dataset_dict[problem].map(
            lambda x: add_dummy_features_to_dataset(x, dummy_features)
        ).repeat()
    return dataset_dict
