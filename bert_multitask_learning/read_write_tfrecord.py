import json
import logging
import os
from functools import partial
from glob import glob
from multiprocessing import cpu_count

import numpy as np
import tensorflow as tf
from joblib import Parallel, delayed

from .bert_preprocessing.create_bert_features import create_bert_features, create_multimodal_bert_features
from .special_tokens import BOS_TOKEN, EOS_TOKEN, EVAL, TRAIN


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

            feature_desc['{}_shape'.format(
                feature_name)] = 'int64'

            # this seems not a good idea
            if len(feature.shape) > 1:
                feature_desc['{}_shape_value'.format(feature_name)] = [
                    None] + list(feature.shape[1:])
            else:
                feature_desc['{}_shape_value'.format(feature_name)] = [
                    None for _ in feature.shape]

        elif np.issubdtype(type(feature), np.float):
            features_tuple[feature_name] = _float_feature(feature)
            features_tuple['{}_shape'.format(
                feature_name)] = _int64_list_feature([])
            feature_desc[feature_name] = 'float32'

            feature_desc['{}_shape'.format(
                feature_name)] = 'int64'
            feature_desc['{}_shape_value'.format(feature_name)] = []
        elif np.issubdtype(type(feature), np.integer):
            features_tuple[feature_name] = _int64_feature(feature)
            features_tuple['{}_shape'.format(
                feature_name)] = _int64_list_feature([])
            feature_desc[feature_name] = 'int64'
            feature_desc['{}_shape'.format(
                feature_name)] = 'int64'
            feature_desc['{}_shape_value'.format(feature_name)] = []
        else:
            if isinstance(feature, str):
                feature = feature.encode('utf8')
            features_tuple[feature_name] = _bytes_feature(feature)
            features_tuple['{}_shape'.format(
                feature_name)] = _int64_list_feature([])
            feature_desc[feature_name] = 'string'
            feature_desc['{}_shape'.format(
                feature_name)] = 'int64'
            feature_desc['{}_shape_value'.format(feature_name)] = []

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


def write_single_problem_chunk_tfrecord(problem,
                                        inputs_list,
                                        target_list,
                                        label_encoder,
                                        params,
                                        tokenizer,
                                        mode):

    def _make_single_problem_data_list(problem, inputs_list, target_list, label_encoder):
        problem_type = params.problem_type[problem]

        # whether this problem is sequential labeling
        # for sequential labeling, targets needs to align with any
        # change of inputs
        is_seq = problem_type in ['seq_tag']
        try:
            example_list = list(zip(inputs_list, target_list))
        except TypeError:
            # target_list is None
            example_list = inputs_list
        # split data_list by shards_per_file
        data_shards = []
        for i in range(0, len(example_list), 10000):
            data_shards.append(example_list[i:i + 10000])

        if isinstance(inputs_list[0], dict) and 'a' not in inputs_list[0]:
            part_fn = partial(create_multimodal_bert_features, problem=problem,
                              label_encoder=label_encoder,
                              params=params,
                              tokenizer=tokenizer,
                              mode=mode,
                              problem_type=problem_type,
                              is_seq=is_seq)
        else:
            part_fn = partial(create_bert_features, problem=problem,
                              label_encoder=label_encoder,
                              params=params,
                              tokenizer=tokenizer,
                              mode=mode,
                              problem_type=problem_type,
                              is_seq=is_seq)
        data_list = Parallel(min(cpu_count(), len(data_shards)))(delayed(part_fn)(example_list=d_list)
                                                                 for d_list in data_shards)
        return data_list

    # single problem in problem_chunk
    if isinstance(problem, str) or (isinstance(problem, list) and len(problem) == 1):
        if isinstance(problem, list):
            problem = problem[0]
        data_list = _make_single_problem_data_list(
            problem, inputs_list=inputs_list, target_list=target_list, label_encoder=label_encoder)

    # multiple problem in problem chunk
    else:
        assert (type(problem), type(inputs_list),
                type(target_list)) == (list, dict, dict)
        # convert data_list to dataframe and join by input_ids
        data_dict = {}
        column_list = []
        for pro in problem:
            data_shards = _make_single_problem_data_list(
                pro, inputs_list=inputs_list[pro], target_list=target_list[pro], label_encoder=label_encoder[pro])
            data_dict[pro] = [
                item for sublist in data_shards for item in sublist]
            column_list.append(list(data_dict[pro][0].keys()))

        # get intersection and use as ensure features are the same
        join_key = list(set(column_list[0]).intersection(*column_list[1:]))

        flat_data_list = []
        while data_dict[problem[0]]:
            d = {}
            for pro in data_dict:
                if not d:
                    d = data_dict[pro].pop(0)
                else:
                    for k in join_key:
                        assert d[k] == data_dict[pro][0][k], 'At iteration {}, feature {} not align. Expected {}, got: {}'.format(
                            len(flat_data_list), k, d[k], data_dict[pro][0][k]
                        )
                    d.update(data_dict[pro].pop(0))
            flat_data_list.append(d)

        data_list = []
        for i in range(0, len(flat_data_list), 10000):
            data_list.append(flat_data_list[i:i + 10000])

    if isinstance(problem, list):
        problem = '_'.join(sorted(problem))

    make_tfrecord(data_list=data_list,
                  output_dir=params.tmp_file_dir,
                  serialize_fn=serialize_fn,
                  prefix=problem,
                  mode=mode)


def write_tfrecord(params, replace=False):
    """Write TFRecord for every problem chunk

    Output location: params.tmp_file_dir

    Arguments:
        params {params} -- params

    Keyword Arguments:
        replace {bool} -- Whether to replace existing tfrecord (default: {False})
    """

    read_data_fn_dict = params.read_data_fn
    path_list = []
    for problem_list in params.problem_chunk:
        # if only one problem in problem chunk, create individual tf record
        if len(problem_list) == 1:
            problem = problem_list[0]
            read_fn = read_data_fn_dict[problem]
            file_dir = os.path.join(params.tmp_file_dir, problem)
            if not os.path.exists(file_dir) or replace:
                read_fn(params, TRAIN)
                read_fn(params, EVAL)
        # if more than one problem in problem chunk, that means multiple
        # same feature space problems are chained by &. In this case, we
        # need to aggregate data from these problems first, then write to one
        # tf record.
        else:
            problem_str = '_'.join(sorted(problem_list))
            file_dir = os.path.join(params.tmp_file_dir, problem_str)
            if not os.path.exists(file_dir) or replace:
                for mode in [TRAIN, EVAL]:

                    input_list_dict = {}
                    target_list_dict = {}
                    label_encoder_dict = {}
                    for p_idx, p in enumerate(problem_list):

                        res_dict = read_data_fn_dict[p](
                            params=params, mode=mode, get_data_num=False, write_tfrecord=False)
                        if p_idx == 0:
                            tokenizer = res_dict['tokenizer']

                        input_list_dict[p] = res_dict['inputs_list']
                        target_list_dict[p] = res_dict['target_list']
                        label_encoder_dict[p] = res_dict['label_encoder']

                    write_single_problem_chunk_tfrecord(
                        problem=problem_list,
                        inputs_list=input_list_dict,
                        target_list=target_list_dict,
                        label_encoder=label_encoder_dict,
                        params=params,
                        tokenizer=tokenizer,
                        mode=mode
                    )


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
        dummy_features.update({k: tf.cast(tf.constant(shape=[1 if s is None else s for s in feature_desc_dict.get('{}_shape_value'.format(k), [])], value=0), v)
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
    for problem_list in params.problem_chunk:
        problem = '_'.join(sorted(problem_list))
        file_dir = os.path.join(params.tmp_file_dir, problem)
        tfrecord_path_list = glob(os.path.join(
            file_dir, '{}_*.tfrecord'.format(mode)))
        feature_desc_dict = json.load(
            open(os.path.join(file_dir, '{}_feature_desc.json'.format(mode))))
        all_feature_desc_dict.update(feature_desc_dict)
        feature_desc = make_feature_desc(feature_desc_dict)
        dataset = tf.data.TFRecordDataset(
            tfrecord_path_list, num_parallel_reads=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(lambda x: tf.io.parse_single_example(
            x, feature_desc), num_parallel_calls=tf.data.experimental.AUTOTUNE).map(
                reshape_tensors_in_dataset, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(
            lambda x: set_shape_for_dataset(x, feature_desc_dict),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        for p in problem_list:
            dataset = dataset.map(lambda x: add_loss_multiplier(x, p),
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset_dict[problem] = dataset

    # add dummy features
    dummy_features = get_dummy_features(dataset_dict, all_feature_desc_dict)
    for problem_list in params.problem_chunk:
        problem = '_'.join(sorted(problem_list))
        dataset_dict[problem] = dataset_dict[problem].map(
            lambda x: add_dummy_features_to_dataset(x, dummy_features),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        ).repeat()
    return dataset_dict
