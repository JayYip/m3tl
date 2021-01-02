import os

import bert_multitask_learning
import numpy as np
import tensorflow as tf

from .test_base import TestBase


class ReadWriteTFRecordTest(TestBase):
    test_features = {
        'int_scalar': 1,
        'float_scalar': 2.0,
        'int_array': [1, 2, 3],
        'float_array': np.array([4, 5, 6], dtype='float32'),
        'int_matrix': [[1, 2, 3], [4, 5, 6]],
        'float_matrix': np.random.uniform(size=(32, 5, 5)),
        'string': 'this is test'
    }

    def setUp(self) -> None:
        super().setUp()
        ser_str = bert_multitask_learning.serialize_fn(
            features=self.test_features, return_feature_desc=False)
        self.example = tf.train.Example()
        self.example.ParseFromString(ser_str)

    def test_serialize_fn(self):

        expected_desc = {'int_scalar': 'int64', 'int_scalar_shape': 'int64', 'int_scalar_shape_value': [],
                         'float_scalar': 'float32', 'float_scalar_shape': 'int64', 'float_scalar_shape_value': [],
                         'int_array': 'int64', 'int_array_shape_value': [None], 'int_array_shape': 'int64',
                         'float_array': 'float32', 'float_array_shape_value': [None], 'float_array_shape': 'int64',
                         'int_matrix': 'int64', 'int_matrix_shape_value': [None, 3], 'int_matrix_shape': 'int64',
                         'float_matrix': 'float32', 'float_matrix_shape_value': [None, 5, 5], 'float_matrix_shape': 'int64',
                         'string': 'string', 'string_shape': 'int64', 'string_shape_value': []}
        ser_str, feat_desc = bert_multitask_learning.serialize_fn(
            features=self.test_features, return_feature_desc=True)
        self.assertEqual(feat_desc, expected_desc)

        example = tf.train.Example()
        example.ParseFromString(ser_str)
        self.assertEqual(
            example.features.feature['int_array'].int64_list.value, [1, 2, 3])

    def test_make_tfrecord(self):
        bert_multitask_learning.make_tfrecord(
            [self.test_features], output_dir=self.tmpfiledir, serialize_fn=bert_multitask_learning.serialize_fn)
        self.assertTrue(os.path.exists(os.path.join(
            self.tmpfiledir, 'train_feature_desc.json')))
        self.assertTrue(os.path.exists(os.path.join(
            self.tmpfiledir, 'train_00000.tfrecord')))

    def test_write_single_problem_chunk_tfrecord(self):
        pass

    def test_write_single_problem_gen_tfrecord(self):
        pass

    def test_write_tfrecord(self):
        self.params.assign_problem(
            'weibo_ner&weibo_fake_cls|weibo_fake_multi_cls|weibo_masklm')
        bert_multitask_learning.write_tfrecord(
            params=self.params)
        self.assertTrue(os.path.exists(os.path.join(
            self.tmpfiledir, 'weibo_fake_cls_weibo_ner')))
        self.assertTrue(os.path.exists(os.path.join(
            self.tmpfiledir, 'weibo_fake_multi_cls')))

    def test_get_dummy_features(self):
        pass

    def test_add_dummy_features_to_dataset(self):
        pass

    def test_read_tfrecord(self):
        self.params.assign_problem(
            'weibo_ner&weibo_fake_cls|weibo_fake_multi_cls|weibo_masklm')
        bert_multitask_learning.write_tfrecord(
            params=self.params, replace=False)
        dataset_dict = bert_multitask_learning.read_tfrecord(
            params=self.params, mode=bert_multitask_learning.TRAIN)
        dataset: tf.data.Dataset = dataset_dict['weibo_fake_cls_weibo_ner']
        self.assertEqual(sorted(list(dataset.element_spec.keys())),
                         ['image_input',
                          'image_mask',
                          'image_segment_ids',
                          'input_ids',
                          'input_mask',
                          'masked_lm_ids',
                          'masked_lm_positions',
                          'masked_lm_weights',
                          'segment_ids',
                          'weibo_fake_cls_label_ids',
                          'weibo_fake_cls_loss_multiplier',
                          'weibo_fake_multi_cls_label_ids',
                          'weibo_fake_multi_cls_loss_multiplier',
                          'weibo_masklm_loss_multiplier',
                          'weibo_ner_label_ids',
                          'weibo_ner_loss_multiplier'])
        # make sure loss multiplier is correct
        ele = next(dataset.as_numpy_iterator())
        self.assertEqual(ele['weibo_fake_cls_loss_multiplier'], 1)
        self.assertEqual(ele['weibo_ner_loss_multiplier'], 1)
        self.assertEqual(ele['weibo_fake_multi_cls_loss_multiplier'], 0)

        # multimodal dataset
        dataset: tf.data.Dataset = dataset_dict['weibo_fake_multi_cls']
        _ = next(dataset.as_numpy_iterator())
